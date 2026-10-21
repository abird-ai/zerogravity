# ROCm HIP runtime

Runtime that runs on ROCm HIP supported AMD GPUs.

Matrix multiplication acceleration is based on RDNA3 by default and requires a ROCm WMMA library for other architectures.
Note that kernel compilation time with that library might be slow.

## RDNA 2

RDNA 2 should work with cubecl-hip, however due RDNA2 not officially being supported by AMD, errors might be encountered. In order for an RDNA 2 GPU to work, the environment variable `HSA_OVERRIDE_GFX_VERSION` must be set to `10.3.0`.

It is known that some errors might occur during initial startup with an RDNA 2 GPU, but if the programs gets through startup then no further errors should be encountered. Re-run the program until it works. This is mostly due to a limitation of the support from AMD.
