import cuda.tile as ct
import cupy as cp
import torch
import random #for random M,K size
''' 
TODO: 
Write a cuTile kernel that reduces a 2D input matrix of arbitrary shape (M, K) 
along its last dimension (K), producing a 1D output vector of shape (M,) 
that contains the per-row sum


Requirements:
- Use either ct.reduce or ct.sum inside the kernel
- Parallelize over the M dimension via the grid
- Verify correctness by comparing the result to torch.sum(mat, dim=1) via torch.allclose.
- Since the tiles can only have dimension sizes that are powers of 2,
  zero-padding inside the kernel can be necessary based on the provided matrix shape
'''


# from the View of Block!!!
@ct.kernel
def reduce(i_matrix, o_vector, k, tile_size: ct.Constant[int]):
    # ct.bid(axes): Returns the index of current block.
    pid = ct.bid(0) #possible values are 0, 1, 2
    
    # ct.load(): loads a tile from Global memory into a tile variable
    i_tile = ct.load(i_matrix, index=(pid,0), shape=(1, tile_size))

# creates a tile with index values from 0 to tile_size - 1
    offsets = ct.arange(tile_size, dtype=ct.int32) 

    # Threads with index < k True, > k False
    mask = ct.where(offsets < k, 1.0, 0.0)

    # everything from index k = 0 
    clean_tile = i_tile * mask

    # If x is a single tile, then the function must take two 0d tile arguments
    # and return the combined 0d tile.
    result = ct.sum(clean_tile, axis =1)
    result_fp16 = result.astype(ct.float16)

    #store result
    ct.store(o_vector,
            index=pid,
            tile=result_fp16)

def next_power_of_2(n):
    if n <= 0: return 1
    # n.bit_length() - number of bits for encoding a number
    return 1 << (n - 1).bit_length()


# ------------------------- MAIN

# set arbitrary shape (M,K)
m = random.randint(0,20)
k = random.randint(0,20)

# round up to the next power of 2
tile_size = next_power_of_2(k)

#reduced_dim = ct.Constant[k] #for cuda.tile.reduce function 
# "axis (int) – an integer constant that specifies the axis to reduce along"

# torch.rand: Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
i_matrix = torch.rand ((m,k), 
                       dtype=torch.float16,
                       device='cuda')
o_vector = torch.zeros((m,),
                        dtype=torch.float16,
                       device='cuda')
print("Created matrix: ", i_matrix, "\n with dimensions ",m,"*",k)

# ---- Parallelize over the M dimension via the grid
grid = (m,1,1) # must be a tuple!!! = number of Blocks, each block gets 1 row



ct.launch(torch.cuda.current_stream().cuda_stream,
                                grid,
                                reduce,
                                (i_matrix, o_vector, k, tile_size))

print(o_vector)
    
