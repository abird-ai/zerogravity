//! Same generated source-numeric argmax factories as launch_numeric_zero.
use super::*;
use cubecl::prelude::BufferCompilationArg;
use cubecl_server::kernel::CubeKernel;

pub(crate) fn numeric_zero_kernels(
    client: &Client,
    vocabulary: usize,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    let mut partials = program_kernels::argmax_partial_count(vocabulary, TILE)?;
    let settings = || {
        KernelSettings::new(
            CubeDim::new_1d(32).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
    };
    let properties = client.properties_shared();
    let target = client.target_properties_shared();
    let buffer = || BufferCompilationArg { inplace: None };
    let mut kernels: Vec<Box<dyn CubeKernel>> = vec![Box::new(
        program_kernels::argmax_logits_numeric_zero::ArgmaxLogitsNumericZero::new(
            settings(),
            properties.clone(),
            target.clone(),
            buffer(),
            buffer(),
            buffer(),
            (),
            (),
            vocabulary,
            TILE,
        ),
    )];
    while partials > 1 {
        kernels.push(Box::new(
            program_kernels::argmax_partials_numeric_zero::ArgmaxPartialsNumericZero::new(
                settings(),
                properties.clone(),
                target.clone(),
                buffer(),
                buffer(),
                buffer(),
                (),
                (),
                partials,
                vocabulary,
                TILE,
            ),
        ));
        partials = program_kernels::argmax_partial_count(partials, TILE)?;
    }
    kernels.push(Box::new(program_kernels::finish_argmax::FinishArgmax::new(
        settings(),
        properties,
        target,
        buffer(),
        buffer(),
        buffer(),
        (),
        vocabulary,
    )));
    Ok(kernels)
}
