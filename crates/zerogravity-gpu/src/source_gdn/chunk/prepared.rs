//! Native definitions for the shared chunk primitive, independent of captured calls.
use cubecl::prelude::{AddressType, BufferCompilationArg, ExecutionMode, KernelSettings};
use cubecl::{client::Client, prelude::CubeDim};
use cubecl_server::kernel::CubeKernel;
use zerogravity_core::Result;
pub(crate) fn kernels(
    client: &Client,
    heads: usize,
    guarded_triangular: bool,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    kernels_with_dense(client, heads, guarded_triangular, false, false)
}
pub(crate) fn kernels_with_dense(
    client: &Client,
    heads: usize,
    guarded_triangular: bool,
    native_products: bool,
    native_solve: bool,
) -> Result<Vec<Box<dyn CubeKernel>>> {
    super::execution::Geometry::new(1, heads)?;
    let mut output: Vec<Box<dyn CubeKernel>> = Vec::new();
    let b = || BufferCompilationArg { inplace: None };
    macro_rules! kernel {
        ($path:path, $($arg:expr),* $(,)?) => {
            output.push(Box::new(<$path>::new(
                KernelSettings::new(CubeDim::new_1d(32).into(),ExecutionMode::Unchecked,AddressType::U32),
                client.properties_shared(),client.target_properties_shared(),$($arg),*
            )));
        };
    }
    kernel!(
        super::normalize::Normalize,
        b(),
        b(),
        b(),
        b(),
        b(),
        (),
        heads
    );
    kernel!(
        super::beta_vectors::BetaVectors,
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        (),
        heads
    );
    kernel!(super::cumulative::Cumulative, b(), b(), b(), b(), (), heads);
    kernel!(
        super::decays::Decays,
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        heads
    );
    if native_products {
        kernel!(super::gram_epilogue::GramEpilogue, b(), b(), b(), b());
    } else {
        kernel!(super::gram::Gram, b(), b(), b(), b(), b(), b(), b(), heads);
    }
    kernel!(
        super::decay_vectors::DecayVectors,
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        b(),
        heads
    );
    if native_solve {
        kernel!(
            crate::program_kernels::copy_strided_words::CopyStridedWords,
            b(),
            b(),
            b(),
            (),
            (),
            (),
            (),
            heads,
            8192usize,
            true
        );
        kernel!(
            crate::program_kernels::certify_finite_strided::CertifyFiniteStrided,
            b(),
            b(),
            (),
            (),
            heads,
            8192usize
        );
    } else if guarded_triangular {
        kernel!(
            super::triangular_guarded::TriangularGuarded,
            b(),
            b(),
            b(),
            b(),
            b(),
            b(),
            heads
        );
    } else {
        kernel!(
            super::triangular::Triangular,
            b(),
            b(),
            b(),
            b(),
            b(),
            b(),
            heads
        );
    }
    if native_products {
        kernel!(super::prior_epilogue::PriorEpilogue, b(), b(), b(), b());
    } else {
        kernel!(
            super::prior_products::PriorProducts,
            b(),
            b(),
            b(),
            b(),
            b(),
            b(),
            b(),
            heads
        );
    }
    if native_products {
        kernel!(
            super::core_epilogue::CoreEpilogue,
            b(),
            b(),
            b(),
            b(),
            (),
            heads
        );
    } else {
        kernel!(
            super::core_product::CoreProduct,
            b(),
            b(),
            b(),
            b(),
            b(),
            b(),
            (),
            heads
        );
    }
    if native_products {
        kernel!(super::state_epilogue::StateEpilogue, b(), b(), b(), b());
    } else {
        kernel!(
            super::state_product::StateProduct,
            b(),
            b(),
            b(),
            b(),
            b(),
            b(),
            heads
        );
    }
    Ok(output)
}
