import torch
from torch.autograd import Function

from mxfp8Quant._C import mxfp8_quantizer


def mxfp8_quantization_func(input, exp, man, bias=None, stochastic=False, clip=True):
    """
    Quantize input tensor into MXFP8 format.

    Args:
        input (torch.Tensor): Input tensor to quantize.
        exp (int): Number of exponent bits.
        man (int): Number of mantissa bits.
        bias (int, optional): Exponent bias. If None, it's computed as 2^(exp-1) - 1.
        stochastic (bool): If True, use stochastic rounding.
        clip (bool): If True, clip input to the MXFP8 representable range.

    Returns:
        torch.Tensor: Quantized tensor.
    """
    assert exp + man == 7, "Sum of exponent and mantissa bits should be 7 for MXFP8."
    assert exp <= 4, "Maximum exponent bits = 4."
    assert exp > 0, "Minimum exponent bits = 1."
    assert man >= 0, "Minimum mantissa bits = 0."

    # Handle half-precision and bfloat16 cases
    is_half = False
    is_bfloat = False
    if input.dtype == torch.half:
        input = input.float()
        is_half = True

    if input.dtype == torch.bfloat16:
        input = input.float()
        is_bfloat = True

    # Default exponent bias
    if bias is None:
        bias = 2 ** (exp - 1) - 1

    # Clip input to representable range in MXFP8
    if clip:
        mxFp = (2 - (2 ** (-1 * man))) * (2 ** ((2**exp) - 1 - bias))  # No Inf max
        input = torch.clamp(input, -mxFp, mxFp)

    # Perform quantization using MXFP8 kernel
    out = mxfp8_quantizer(input, exp, man, bias, stochastic)

    # Convert back to original dtype if needed
    if is_half:
        out = out.half()
    if is_bfloat:
        out = out.bfloat16()
    return out
