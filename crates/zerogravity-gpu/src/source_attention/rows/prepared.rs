//! Cold capacity kernels; invocation length/base never select a compiled body.
use super::*;
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;

pub(crate) fn kernels(client: &Client, g: Geometry) -> Vec<Box<dyn CubeKernel>> {
    let mut kernels: Vec<Box<dyn CubeKernel>> = Vec::with_capacity(8);
    let b = || BufferCompilationArg { inplace: None };
    macro_rules! add {
        ($kind:path,$lanes:expr,$($arg:expr),* $(,)?)=> {
            kernels.push(Box::new(<$kind>::new(KernelSettings::new(CubeDim::new_1d($lanes).into(),ExecutionMode::Unchecked,AddressType::U32),
                client.properties_shared(),client.target_properties_shared(),$($arg),*)))
        }
    }
    add!(
        crate::source_attention::kernels::qk::Qk,
        64,
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        (),
        (),
        (),
        256usize,
        g.heads,
        g.kv_heads,
        g.capacity
    );
    add!(mask::Mask, 32, b(), (), g.capacity);
    for maximum in [true, false] {
        add!(
            crate::source_attention::kernels::reduce::Reduce,
            32,
            b(),
            b(),
            b(),
            (),
            maximum,
            g.capacity
        );
    }
    add!(
        crate::source_attention::kernels::exponentials::Exponentials,
        32,
        b(),
        b(),
        b(),
        b(),
        (),
        g.heads,
        g.capacity
    );
    add!(
        crate::source_attention::kernels::inverse::Inverse,
        32,
        b(),
        b(),
        b(),
        g.heads
    );
    add!(
        crate::source_attention::kernels::probabilities::Probabilities,
        32,
        b(),
        b(),
        b(),
        b(),
        (),
        g.heads,
        g.capacity
    );
    add!(
        crate::source_attention::kernels::pv::Pv,
        32,
        b(),
        b(),
        b(),
        b(),
        b(),
        (),
        (),
        (),
        256usize,
        g.heads,
        g.kv_heads,
        g.capacity
    );
    kernels
}
