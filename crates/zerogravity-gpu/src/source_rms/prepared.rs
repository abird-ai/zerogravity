//! Original RMS kernel constructors, shared by source-call preparation.
use super::*;
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;

pub(crate) fn kernels(
    client: &Client,
    width: usize,
    groups: usize,
    mode: Gamma,
) -> Vec<Box<dyn CubeKernel>> {
    let mut out: Vec<Box<dyn CubeKernel>> = Vec::new();
    let b = || BufferCompilationArg { inplace: None };
    macro_rules! add {
        ($k:path,$($a:expr),* $(,)?)=>{out.push(Box::new(<$k>::new(
            KernelSettings::new(CubeDim::new_1d(32).into(),ExecutionMode::Unchecked,AddressType::U32),
            client.properties_shared(),client.target_properties_shared(),$($a),*
        )))}
    }
    let count = width * groups;
    add!(super::kernels::squares::Squares, b(), b(), b(), count);
    add!(super::kernels::sum::Sum, b(), b(), b(), width);
    for stage in 0..4u32 {
        add!(
            super::kernels::scalar::Scalar,
            b(),
            b(),
            b(),
            stage,
            width,
            groups
        );
    }
    add!(
        super::kernels::normalize::Normalize,
        b(),
        b(),
        b(),
        b(),
        width,
        count
    );
    add!(
        super::kernels::weight::Weight,
        b(),
        b(),
        b(),
        b(),
        width,
        count,
        mode.mode()
    );
    if mode != Gamma::DirectF32AfterBf16 {
        add!(super::kernels::store::Store, b(), b(), b(), count);
    }
    out
}
