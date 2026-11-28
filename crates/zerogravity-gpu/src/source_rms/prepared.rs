//! Original RMS kernel constructors, shared by text-invocation preparation.
use super::*;
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;

pub(crate) fn kernels(
    client: &Client,
    width: usize,
    groups: usize,
    mode: Gamma,
    residual: bool,
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
    if residual {
        add!(
            super::kernels::residual_squares::ResidualSquares,
            b(),
            b(),
            b(),
            b(),
            b()
        );
    } else {
        add!(super::kernels::squares::Squares, b(), b(), b());
    }
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

pub(crate) fn region_kernels(
    client: &Client,
    width: usize,
    mode: Gamma,
    residual: bool,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    super::scratch_words(width, 1)?;
    let settings = || {
        KernelSettings::new(
            CubeDim::new_1d(32).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
    };
    let b = || BufferCompilationArg { inplace: None };
    let p = client.properties_shared();
    let t = client.target_properties_shared();
    let inverse: Box<dyn CubeKernel> = if residual {
        Box::new(
            super::kernels::region_residual_inverse::RegionResidualInverse::new(
                settings(),
                p.clone(),
                t.clone(),
                b(),
                b(),
                b(),
                b(),
                b(),
                b(),
                width,
            ),
        )
    } else {
        Box::new(super::kernels::region_inverse::RegionInverse::new(
            settings(),
            p.clone(),
            t.clone(),
            b(),
            b(),
            b(),
            b(),
            width,
        ))
    };
    Ok(vec![
        inverse,
        Box::new(
            super::kernels::normalize_weight_store::NormalizeWeightStore::new(
                settings(),
                p,
                t,
                b(),
                b(),
                b(),
                b(),
                b(),
                width,
                mode.mode(),
            ),
        ),
    ])
}
