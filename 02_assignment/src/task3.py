import cuda.tile as ct
import cupy as cp
import torch

'''
a) TODO: write a cuTile kernel that adds 2 4D tensors A and B element-wise 
and stores the result in C. All tensors have shape (M, N, K, L).

Implement the kernel twice using the following two approaches:

1. Each kernel program is responsible for computing the sum of a 2D output 
tile that covers dimensions K and L
2. Each kernel program is responsible for computing the sum of a 2D output
 tile that covers dimensions M and N

Parallelize over the remaining two dimensions in each respective case.

Verify both implementations against PyTorch's native A + B via torch.allclose

b) Benchmark both of your kernels and report the average runtimes 
'''

@ct.kernel
def sum_kl (a_tensor, b_tensor,  c_tensor, 
            k: ct.Constant[int],
            l: ct.Constant[int]):
    
    pid_m = ct.bid(0)   # index on M-axis
    pid_n = ct.bid(1)   # index on N-axis

    a_tile = ct.load(a_tensor, index = (pid_m, pid_n, 0, 0), shape = (1,1,k,l))
    b_tile = ct.load(b_tensor, index = (pid_m, pid_n, 0, 0), shape = (1,1,k,l))
    
    #c_tile = a_tile + b_tile
    c_tile = ct.add(a_tile, b_tile)
    
    ct.store(c_tensor,  (pid_m, pid_n, 0, 0), tile=c_tile)


@ct.kernel
def sum_mn (a_tensor, b_tensor,  c_tensor, 
            m: ct.Constant[int],
            n: ct.Constant[int]):

    pid_k = ct.bid(0)   # index on K-axis
    pid_l = ct.bid(1)   # index on L-axis

    a_tile = ct.load(a_tensor, index = (0,0, pid_k, pid_l), shape = (m,n,1,1))
    b_tile = ct.load(b_tensor, index = (0,0,pid_k, pid_l), shape = (m,n,1,1))
    
    c_tile = ct.add(a_tile, b_tile)
    
    ct.store(c_tensor,  (0,0, pid_k, pid_l), tile=c_tile)


def run_kernels():
    # set dimension sizes
    m, n, k, l = 16, 128, 16, 128

    # torch.rand: Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
    a_tensor = torch.rand ((m,n,k,l), 
                        dtype=torch.float16,
                        device='cuda')
    b_tensor = torch.rand ((m,n,k,l), 
                        dtype=torch.float16,
                        device='cuda')

    c_tensor = torch.zeros((m,n,k,l),
                            dtype=torch.float16,
                        device='cuda')

    # Kernel 1
    ct.launch(torch.cuda.current_stream().cuda_stream,
                                    (m,n,1), # grid size: parallelize over M,N
                                    sum_kl,
                                    (a_tensor, b_tensor, c_tensor, k,l))

    print("1: is correct: ", torch.allclose(c_tensor, a_tensor + b_tensor))

    c_tensor.zero_()

    # Kernel 2
    ct.launch(torch.cuda.current_stream().cuda_stream,
                                    (k,l,1), # grid size: parallelize over K,L
                                    sum_mn,
                                    (a_tensor, b_tensor, c_tensor, m, n))


    print("2: is correct: ", torch.allclose(c_tensor, a_tensor + b_tensor))



if __name__ == "__main__":
    run_kernels()