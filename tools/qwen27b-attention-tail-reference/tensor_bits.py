"""Bounded logical raw-bit transport only; no tensor/provider import on import."""
import sys
from contract import require


def tensor_bits(value, torch):
    require(sys.byteorder == 'little', 'pinned little-endian raw-bit transport')
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.layout == torch.strided, 'raw-bit transport requires a CPU strided tensor')
    require(value.dtype in (torch.bfloat16, torch.float32),
            'raw-bit transport only original BF16/F32')
    require(value.numel()*value.element_size() <= 128*1024**2,
            'bounded raw-bit transport before view/copy')
    integer = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
    # Equal element size keeps singleton/nonunit strides legal. NumPy writes
    # logical C order; no floating conversion or source/storage mutation.
    raw = value.detach().view(integer).numpy().tobytes(order='C')
    require(len(raw) == value.numel()*value.element_size(), 'complete logical raw-bit transport')
    return raw
