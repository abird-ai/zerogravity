//! Explicit listener ownership. No environment, signals or runtime shutdown.
use crate::{
    Error, Result, TextService, auth, http,
    ingress::IngressLimits,
    metrics::{Counters, ServerMetrics},
};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};
use zerogravity_engine::{
    Runtime,
    router::pending::{self, ServingCharge},
    state_cache::PrefixSharingScope,
};

#[cfg(test)]
mod runtime_tests;
#[cfg(test)]
mod tests;

pub struct ServerBuilder {
    service: Arc<dyn TextService>,
    address: SocketAddr,
    limits: IngressLimits,
    api_key: Option<String>,
    enable_per_request_metrics: bool,
    prefix_sharing_scope: Option<PrefixSharingScope>,
}
pub struct Server {
    builder: ServerBuilder,
}
/// Dropping the handle stops ingress, interrupts sockets and joins every handler.
/// The caller's runtime and unrelated SDK tasks remain open.
pub struct ServerHandle {
    address: SocketAddr,
    worker: Option<JoinHandle<std::io::Result<()>>>,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<Counters>,
    // Retained through the accept-thread join, including panic/error paths.
    _charge: ServingCharge,
}
struct AcceptCaptures {
    listener: TcpListener,
    service: Arc<dyn TextService>,
    connections: http::Connections,
    config: http::ServeConfig,
    metrics: Arc<Counters>,
}
impl AcceptCaptures {
    fn run(self) -> std::io::Result<()> {
        http::serve(
            self.listener,
            self.service,
            self.connections,
            self.config,
            self.metrics,
        )
    }
}
impl ServerBuilder {
    pub fn new(runtime: impl Into<Runtime>) -> Self {
        Self::from_service(Arc::new(runtime.into()))
    }
    /// Also used by adapters during migration; no second scheduler is created.
    pub fn from_service(service: Arc<dyn TextService>) -> Self {
        Self {
            service,
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080),
            limits: IngressLimits::default(),
            api_key: None,
            enable_per_request_metrics: false,
            prefix_sharing_scope: None,
        }
    }
    pub fn bind(mut self, address: SocketAddr) -> Self {
        self.address = address;
        self
    }
    pub fn limits(mut self, limits: IngressLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Explicit optional bearer credential. Never read from the environment.
    pub fn api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }
    pub fn enable_per_request_metrics(mut self, enabled: bool) -> Self {
        self.enable_per_request_metrics = enabled;
        self
    }
    /// Trusted server-owned authorization scope for cross-request prefix reuse.
    /// HTTP clients cannot supply or override this value.
    pub fn prefix_sharing_scope(mut self, scope: Option<PrefixSharingScope>) -> Self {
        self.prefix_sharing_scope = scope;
        self
    }
    /// Cold validation only: no listener, model load or thread creation.
    pub fn build(self) -> Result<Server> {
        self.service.require_open()?;
        auth::require_loopback(self.address)?;
        self.limits.validate()?;
        auth::validate_key(self.api_key.as_deref())?;
        Ok(Server { builder: self })
    }
}
impl Server {
    pub fn builder(runtime: impl Into<Runtime>) -> ServerBuilder {
        ServerBuilder::new(runtime)
    }
    /// Configured address only; port zero is resolved by `start`.
    pub fn address(&self) -> SocketAddr {
        self.builder.address
    }
    pub fn metrics(&self) -> ServerMetrics {
        ServerMetrics::default()
    }
    pub fn start(self) -> Result<ServerHandle> {
        let builder = self.builder;
        builder.service.require_open()?;
        // Price captured owners before creating shared controls or a thread.
        // Native stacks/TLS/OS mappings remain external unquantified headroom.
        let control = zerogravity_core::resource_arbiter::rust_thread_control_bytes(
            std::alloc::Layout::new::<AcceptCaptures>(),
            Some(("zg-accept".len(), "zg-accept".len())),
        )?;
        let service_layout = std::alloc::Layout::new::<[usize; 2]>()
            .extend(std::alloc::Layout::for_value(builder.service.as_ref()))
            .map_err(|_| Error::Limit("server service allocation layout overflow".into()))?
            .0
            .pad_to_align();
        let owners = pending::checked_sum(&[
            control,
            // The shared std-thread envelope prices a unit return packet.
            // Add the full result row for this accept worker's non-unit return.
            std::mem::size_of::<Option<std::thread::Result<std::io::Result<()>>>>() as u64,
            std::mem::size_of::<ServerHandle>() as u64,
            std::mem::size_of::<(usize, usize, AtomicBool)>() as u64,
            std::mem::size_of::<(usize, usize, Counters)>() as u64,
            service_layout.size() as u64,
            builder
                .api_key
                .as_ref()
                .map_or(0, |key| key.capacity() as u64),
        ])
        .map_err(service_error)?;
        let charge = builder
            .service
            .charge_ingress("http-server", owners)
            .map_err(service_error)?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(Counters::default());
        let connections = http::Connections::new(
            builder.service.as_ref(),
            Arc::clone(&shutdown),
            builder.limits.connections,
        )
        .map_err(io_error)?;
        let listener = TcpListener::bind(builder.address).map_err(io_error)?;
        listener.set_nonblocking(true).map_err(io_error)?;
        let address = listener.local_addr().map_err(io_error)?;
        let captures = AcceptCaptures {
            listener,
            service: builder.service,
            connections,
            config: http::ServeConfig {
                api_key: builder.api_key,
                enable_per_request_metrics: builder.enable_per_request_metrics,
                prefix_sharing_scope: builder.prefix_sharing_scope,
                limits: builder.limits,
            },
            metrics: Arc::clone(&metrics),
        };
        let worker = std::thread::Builder::new()
            .name("zg-accept".into())
            .spawn(move || captures.run())
            .map_err(io_error)?;
        Ok(ServerHandle {
            address,
            worker: Some(worker),
            shutdown,
            metrics,
            _charge: charge,
        })
    }
}
impl ServerHandle {
    /// Actual bound address, including the kernel-selected ephemeral port.
    pub fn local_addr(&self) -> SocketAddr {
        self.address
    }
    pub fn metrics(&self) -> ServerMetrics {
        self.metrics.snapshot()
    }
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }
    /// Wait for stop or listener failure. No runtime lock is held across joins.
    pub fn join(mut self) -> Result<()> {
        self.join_inner()
    }
    pub fn shutdown(mut self) -> Result<()> {
        self.stop();
        self.join_inner()
    }
    fn join_inner(&mut self) -> Result<()> {
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| Error::State("HTTP accept thread panicked".into()))?
                .map_err(io_error)?;
        }
        Ok(())
    }
}
impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop();
        let _ = self.join_inner();
    }
}
fn io_error(error: std::io::Error) -> Error {
    Error::State(error.to_string())
}
fn service_error(error: zerogravity_engine::router::ServiceError) -> Error {
    error.into_core()
}
