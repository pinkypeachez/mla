#L2 Cache Optimization via Block Swizzling

'''
a) implement a swizzled matrix multiplication kernel. 

The requirements are the same as in Task 2, 

Requirements from task2:

Each kernel program is responsible for producing one output tile of shape (m_tile, n_tile)
The kernel should work with tile sizes that are specified by the calling function
The kernel should support matrix shapes that are not powers of 2
Use ct.mma for the inner accumulation step


...except block IDs should NOT be mapped in row-major order!

Swizzle them for L2 cache reuse.  You can assume a contraction dimension size of 4096.

Report how you choose to map the BIDs and why. 
Verify correctness of the swizzled kernel against torch.matmul.
'''

m, n = 8192, 8192
k = 4096 #assume a contraction dimension size of 4096
tile_size = (m,n)  #tile size specified by calling function

# a und b fp16