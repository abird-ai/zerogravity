use crate::{Error, Result, auth, ingress::IngressLimits, metrics::ServerMetrics};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use zerogravity_engine::Runtime;

pub struct ServerBuilder {
    runtime: Runtime,
    address: SocketAddr,
    limits: IngressLimits,
}
pub struct Server {
    builder: ServerBuilder,
}
/// No public constructor: only an actually started server may issue a handle.
pub struct ServerHandle {
    _private: (),
}
impl ServerBuilder {
    pub fn new(runtime: impl Into<Runtime>) -> Self {
        Self {
            runtime: runtime.into(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080),
            limits: IngressLimits::default(),
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
    /// Cold validation only: does not bind, open sources or construct workers.
    pub fn build(self) -> Result<Server> {
        self.runtime.require_open()?;
        auth::require_loopback(self.address)?;
        self.limits.validate()?;
        Ok(Server { builder: self })
    }
}
impl Server {
    pub fn builder(runtime: impl Into<Runtime>) -> ServerBuilder {
        ServerBuilder::new(runtime)
    }
    pub fn address(&self) -> SocketAddr {
        self.builder.address
    }
    pub fn metrics(&self) -> ServerMetrics {
        ServerMetrics::default()
    }
    pub fn start(self) -> Result<ServerHandle> {
        self.builder.runtime.require_open()?;
        Err(Error::Unsupported(
            "Fabric listener/protocol execution awaits F2; existing CLI server unchanged".into(),
        ))
    }
}
impl ServerHandle {
    pub fn shutdown(self) -> Result<()> {
        Err(Error::Unsupported(
            "Fabric server shutdown awaits an actual F2 server handle".into(),
        ))
    }
}
