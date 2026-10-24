//! The existing CPU topology, with every source store kept observable.
use super::*;
impl TypedBody {
    fn certify_stage(&self, b: Buffer) {
        let lease = self.lease(b);
        let n = lease.bytes() / 4;
        unsafe {
            crate::program_kernels::certify_finite_strided::launch_unchecked(
                self.service.client(),
                grid(n),
                threads(),
                arg(&lease.handle(), n),
                arg(&self.lease(Buffer::Status).handle(), 1),
                0,
                n as u32,
                1,
                n,
            );
        }
    }
    pub(super) fn enqueue_rms(&self, cancelled: Option<&AtomicBool>) -> Result<usize> {
        use Buffer::*;
        let h = self.shape.width;
        let client = self.service.client();
        let a = |b: Buffer| {
            let l = self.lease(b);
            arg(&l.handle(), l.bytes() / 4)
        };
        let mut launches = 0;
        macro_rules! stage {
            ($out:expr,$call:expr) => {{
                cancel(cancelled)?;
                self.service.ensure_usable()?;
                unsafe { $call };
                self.certify_stage($out);
                launches += 2;
            }};
        }
        stage!(
            Gamma,
            kernels::widen::launch_unchecked(client, grid(h), threads(), a(GammaRaw), a(Gamma))
        );
        stage!(
            RuntimeGamma,
            kernels::scalar_stage::launch_unchecked(
                client,
                grid(h),
                threads(),
                a(Gamma),
                a(RuntimeGamma),
                5,
                h
            )
        );
        stage!(
            Squares,
            kernels::scalar_stage::launch_unchecked(
                client,
                grid(h),
                threads(),
                a(Input),
                a(Squares),
                0,
                h
            )
        );
        stage!(
            Sum,
            kernels::sum::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                threads(),
                a(Squares),
                a(Zero),
                a(Sum),
                h
            )
        );
        stage!(
            Mean,
            kernels::scalar_stage::launch_unchecked(
                client,
                grid(1),
                threads(),
                a(Sum),
                a(Mean),
                1,
                h
            )
        );
        stage!(
            Denominator,
            kernels::scalar_stage::launch_unchecked(
                client,
                grid(1),
                threads(),
                a(Mean),
                a(Denominator),
                2,
                h
            )
        );
        stage!(
            Sqrt,
            kernels::scalar_stage::launch_unchecked(
                client,
                grid(1),
                threads(),
                a(Denominator),
                a(Sqrt),
                3,
                h
            )
        );
        stage!(
            Inverse,
            kernels::scalar_stage::launch_unchecked(
                client,
                grid(1),
                threads(),
                a(Sqrt),
                a(Inverse),
                4,
                h
            )
        );
        stage!(
            Normalized,
            kernels::apply::launch_unchecked(
                client,
                grid(h),
                threads(),
                a(Input),
                a(Inverse),
                a(Normalized),
                true
            )
        );
        stage!(
            Weighted,
            kernels::apply::launch_unchecked(
                client,
                grid(h),
                threads(),
                a(Normalized),
                a(RuntimeGamma),
                a(Weighted),
                false
            )
        );
        stage!(
            Rms,
            kernels::round_pack::launch_unchecked(
                client,
                grid(h / 2),
                threads(),
                a(Weighted),
                a(Rms),
                a(PackedRms)
            )
        );
        Ok(launches)
    }
}
