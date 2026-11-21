//! Original RMS kernel constructors, shared by text-invocation preparation.
use super::*;
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;

pub(crate) fn kernels(
    client: &Client,
    width: usize,
    groups: usize,
    mode: Gamma,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    super::scratch_words(width, groups)?;
    let mut out: Vec<Box<dyn CubeKernel>> = Vec::new();
    let b = || BufferCompilationArg { inplace: None };
    macro_rules! add {
        ($k:path,$($a:expr),* $(,)?)=>{out.push(Box::new(<$k>::new(
            KernelSettings::new(CubeDim::new_1d(32).into(),ExecutionMode::Unchecked,AddressType::U32),
            client.properties_shared(),client.target_properties_shared(),$($a),*
        )))}
    }
    add!(super::kernels::squares::Squares, b(), b(), b());
    add!(super::kernels::sum::Sum, b(), b(), b(), width);
    for stage in 0..4u32 {
        add!(super::kernels::scalar::Scalar, b(), b(), b(), stage, width);
    }
    add!(
        super::kernels::normalize::Normalize,
        b(),
        b(),
        b(),
        b(),
        width
    );
    add!(
        super::kernels::weight::Weight,
        b(),
        b(),
        b(),
        b(),
        width,
        mode.mode()
    );
    if mode != Gamma::DirectF32AfterBf16 {
        add!(super::kernels::store::Store, b(), b(), b());
    }
    Ok(out)
}
