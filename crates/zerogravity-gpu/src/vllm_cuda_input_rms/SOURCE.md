# Closed input RMS source

Diagnostic only. Actual target: official HF .8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, vLLM `db9527a…`,
Torch `cf30153c4c131c8164ee7798e5022d810682e2cb`, CUDA12.9, SM86.

- Genuine oracle2 records native IR RMS for every actual row count1/64/2:
  BF16 input/raw ordinary gamma, F32 `(gamma+1)` and variance epsilon1e-6.
- Installed `ATen/native/cuda/Reduce.cuh`: `mnt_wrapper<float>`512 threads;
  `setReduceConfig` vectorized input4; `input_vectorized_thread_reduce_impl`
  four accumulators then ordered left merge; `block_x_reduce` shared offsets
  128/64/32 then CUDA warp offsets16/8/4/2/1. No Y/global reduction here.
- Installed `ATen/native/SharedReduceOps.h` `MeanOps`: F32 add, F32 factor1/1024.
  The original binary exports
  `gpu_reduce_kernel<float,float,4,4,MeanOps<float,float,float,float>,double>`
  and `reduce_kernel<512,1,ReduceOp<float,MeanOps<…>,unsigned,float,4,4>>`.
- Native SM86 float rsqrt SASS: `MUFU.RSQ`; a separate denormal rescale branch
  multiplies by2^24 then2^12. Squared mean + epsilon≥1e-6 excludes that branch.
  The inverse stage uses `rsqrt.approx.ftz.f32`, preserving the native atom.
- Each original cast, add, square, mean, inverse and product has its own
  device output and launch. No BF16 cast between normalization and gamma.
  The caller must retain all admitted intermediates until original completion.

The native source/compiled instruction derivation is not a physical result.
Require genuine RMS output bits, then computed projections/GDN/state/output;
no imported intermediate in the final connected gate. No full-model eligibility.
