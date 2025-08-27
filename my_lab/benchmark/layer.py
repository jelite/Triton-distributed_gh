import torch.nn as nn
import torch
import triton
import triton.language as tl


class LinearTorchB16(nn.Module):
    def __init__(self, weight: torch.nn.Module, bias: torch.nn.Module):
        super().__init__()

        self.weight = weight
        self.bias   = bias
   
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_partial = x.matmul(self.weight) + self.bias  # [B, out_per_rank]
        return out_partial  # [B, out_per_rank]
    

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512,  'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256,  'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 1024, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512,  'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512,  'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _bf16_matmul_bias_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # power-of-two only
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # power-of-two only
    offs_k = tl.arange(0, BLOCK_K)                     # power-of-two only

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0).to(tl.bfloat16)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0).to(tl.bfloat16)
        acc += tl.dot(a, b)  # BF16 inputs, FP32 accumulate
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K] (bf16, cuda)
    weight: [K, N] (bf16, cuda)   # non-transposed
    bias:   [N]    (bf16, cuda)
    returns: [M, N] (bf16)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _bf16_matmul_bias_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


class LinearTritonBF16(nn.Module):
    def __init__(self, weight: torch.nn.Module, bias: torch.nn.Module):
        super().__init__()
        self.weight = weight
        self.bias   = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_linear_bf16(x, self.weight, self.bias)