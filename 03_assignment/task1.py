# implement two cuTile kernels that each compute a matrix multiplication 
# $A x B = C$ with `shape(A) = (64, 4096)`, `shape(B) = (4096, 64)` and `shape(C) = (64, 64)`. 
#
# 1. kernel_fp16: A and B are `FP16`, output `C` is `FP32`
# 2. kernel_fp32: A and B are `FP32`, output `C` is `FP32`
# ct.mma

import cuda.tile as ct
import cupy as cp
import torch
import triton



@ct.kernel
def kernel(a, b, acc,
                m: ct.Constant[int],
                iterate: ct.Constant[int]):

    acc_tile = ct.full((m, m), fill_value=0, dtype=ct.float32)

    for i in range(iterate):
        a_tile = ct.load(a, index=(0, i), shape=(m, m))
        b_tile = ct.load(b, index=(i, 0), shape=(m, m))
        acc_tile = ct.mma(a_tile, b_tile, acc_tile)

    ct.store(acc, (0, 0), tile=acc_tile)    


grid = (1,)
m,n = 64, 4096
iterate = n // m
a_fp32 = torch.rand ((m,n), 
                        dtype=torch.float32,
                        device='cuda')
b_fp32 = torch.rand ((n,m), 
                        dtype=torch.float32,
                        device='cuda')

# FP16 per Cast
a_fp16 = a_fp32.half()
b_fp16 = b_fp32.half()

acc = torch.zeros((m,m),
                            dtype=torch.float32,
                        device='cuda')


########################################
torch.cuda.init()

# FLOAT 32
ct.launch(torch.cuda.current_stream(),
            grid,
            kernel,
            (a_fp32,b_fp32,acc, m, iterate))  
  
print("torch.allclose verification: ", torch.allclose(acc, (a_fp32 @ b_fp32).float(), atol=1e-3, rtol=1e-3))

acc = acc.zero_()

#FLOAT 16
ct.launch(torch.cuda.current_stream(),
            grid,
            kernel,
            (a_fp16,b_fp16,acc, m, iterate))    

print("torch.allclose verification: ", torch.allclose(acc, (a_fp16 @ b_fp16).float(), atol=1e-3, rtol=1e-3))


