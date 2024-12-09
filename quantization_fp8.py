import torch
import torch.nn.functional as F

from mxfp8_kernel.mxfp8_operation import mxfp8_quantization_func


def compute_scaling_factor(amax, fp_max):
    scale = torch.ones_like(amax)
    exp = torch.floor(torch.log2(fp_max / amax))
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(exp), sf, scale)
    sf = torch.where(exp < 0, 1 / sf, sf)
    return sf


def mx_quantize(x, exp_bits, stochastic=False):
    output = x.float().clone()
    exp_bias = 7  # MXFP8 has an exponent bias of 7
    man_width = 7 - exp_bits  # Mantissa width for MXFP8
    mxFp = (2 - (2 ** (-1 * man_width))) * (2 ** ((2**exp_bits) - 1 - exp_bias))  # No Inf max
    scale = compute_scaling_factor(output.abs().max(dim=-1, keepdim=True)[0], mxFp)
    output = mxfp8_quantization_func(
        (output * scale), exp=exp_bits, man=man_width, bias=exp_bias, stochastic=stochastic, clip=True
    ) / scale
    return output


class MXFp8QuantLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MXFp8QuantLinear, self).__init__(in_features, out_features, bias)
        self.blocksize = 32

    def forward(self, input):
        return MXFp8QuantLinearFunc.apply(input, self.weight, self.bias)


class MXFp8QuantLinearFunc(torch.autograd.Function):
    """Linear semi-top level module for MXFP8 quantization."""
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        blocksize = 32

        # Input quantization using MXFP8
        qinput = mx_quantize(input.reshape(-1, blocksize), exp_bits=4, stochastic=False)
        qinput = qinput.reshape(input.shape)

        # Weight quantization using MXFP8
        qweight = mx_quantize(weight.reshape(-1, blocksize), exp_bits=4, stochastic=False)
        qweight = qweight.reshape(weight.shape)

        # Compute the linear layer
        output = F.linear(qinput, qweight, bias)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for FP8 quantization.
        Computes gradients for input, weight, and bias using FP8 quantization.
        """
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        blocksize = 32  # Process tensors in blocks of size 32
        grad_output = grad_output.contiguous()

        #### GRAD INPUT ####
        # Quantize weight (RDN 121)
        qos_weight = torch.movedim(weight, -1, -2)  # Rearrange dimensions
        sh = qos_weight.shape
        qos_weight = mx_quantize(qos_weight.reshape(-1, blocksize), exp_bits=4, stochastic=False)
        qos_weight = torch.movedim(qos_weight.reshape(sh), -1, -2)  # Restore original dimensions

        # Quantize grad_output (SR 130)
        qos_grad_output = mx_quantize(grad_output.reshape(-1, blocksize), exp_bits=4, stochastic=True)
        qos_grad_output = qos_grad_output.reshape(grad_output.shape)  # Restore shape

        # Compute grad_input
        grad_input = qos_grad_output.matmul(qos_weight.contiguous())

        #### GRAD WEIGHT ####
        # Quantize input (RDN 121)
        qex_input = torch.movedim(input, -1, -2)  # Rearrange dimensions
        sh = qex_input.shape
        qex_input = mx_quantize(qex_input.reshape(-1, blocksize), exp_bits=4, stochastic=False)
        qex_input = torch.movedim(qex_input.reshape(sh), -1, -2)  # Restore dimensions
        qex_input = qex_input.contiguous().reshape(-1, weight.shape[1])  # Reshape for matmul

        # Quantize grad_output (SR 130)
        qex_grad_output = torch.movedim(grad_output, -1, -2)  # Rearrange dimensions
        sh = qex_grad_output.shape
        qex_grad_output = mx_quantize(qex_grad_output.reshape(-1, blocksize), exp_bits=4, stochastic=True)
        qex_grad_output = torch.movedim(qex_grad_output.reshape(sh), -1, -2)  # Restore dimensions
        qex_grad_output = qex_grad_output.reshape(-1, weight.shape[0])  # Reshape for matmul

        # Compute grad_weight
        grad_weight = qex_grad_output.t().matmul(qex_input)

        #### GRAD BIAS ####
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias

