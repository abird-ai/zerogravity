//! Exact constructors for the shared single-row launch path, not a case catalogue.
use super::*;
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;

pub(crate) fn kernels(
    client: &Client,
    geometry: TorchCpuAttentionGeometry,
    position: usize,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    let (q, kv, tokens) = geometry.validate(position)?;
    let mut kernels: Vec<Box<dyn CubeKernel>> = Vec::with_capacity(18);
    let b = || BufferCompilationArg { inplace: None };
    macro_rules! add {
        ($kind:path, $lanes:expr, $($arg:expr),* $(,)?) => {
            kernels.push(Box::new(<$kind>::new(
                KernelSettings::new(CubeDim::new_1d($lanes).into(), ExecutionMode::Unchecked, AddressType::U32),
                client.properties_shared(), client.target_properties_shared(), $($arg),*
            )))
        };
    }
    add!(
        crate::source_rotary::kernels::angles::Angles,
        32,
        b(),
        b(),
        (),
        32usize
    );
    add!(
        crate::source_trig::sincos_angles::SincosAngles,
        32,
        b(),
        b(),
        b(),
        32usize
    );
    for n in [q, kv] {
        add!(
            crate::program_kernels::copy_strided_words::CopyStridedWords,
            32,
            b(),
            b(),
            b(),
            (),
            (),
            (),
            (),
            1usize,
            n,
            true
        );
        add!(
            crate::source_rotary::kernels::apply::Apply,
            32,
            b(),
            b(),
            b(),
            b(),
            256usize,
            n,
            64usize
        );
    }
    add!(
        crate::source_attention::transport::append::Append,
        32,
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        kv,
        position
    );
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
        geometry.query_heads,
        geometry.kv_heads,
        tokens
    );
    for maximum in [true, false] {
        add!(
            crate::source_attention::kernels::reduce::Reduce,
            32,
            b(),
            b(),
            b(),
            (),
            maximum,
            tokens
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
        geometry.query_heads,
        tokens
    );
    add!(
        crate::source_attention::kernels::inverse::Inverse,
        32,
        b(),
        b(),
        b(),
        geometry.query_heads
    );
    add!(
        crate::source_attention::kernels::probabilities::Probabilities,
        32,
        b(),
        b(),
        b(),
        b(),
        (),
        geometry.query_heads,
        tokens
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
        geometry.query_heads,
        geometry.kv_heads,
        tokens
    );
    add!(
        crate::source_sigmoid::sigmoid_shift::SigmoidShift,
        32,
        b(),
        b(),
        b(),
        q
    );
    add!(
        crate::source_sigmoid::sigmoid::Sigmoid,
        32,
        b(),
        b(),
        b(),
        b(),
        b(),
        q
    );
    add!(
        crate::source_attention::transport::gate::Gate,
        32,
        b(),
        b(),
        b(),
        b(),
        q
    );
    let mut ids = std::collections::HashSet::new();
    kernels.retain(|kernel| ids.insert(kernel.id()));
    Ok(kernels)
}
