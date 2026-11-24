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
    kernel!(super::gram::Gram, b(), b(), b(), b(), b(), b(), b(), heads);
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
    if guarded_triangular {
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
    Ok(output)
}
