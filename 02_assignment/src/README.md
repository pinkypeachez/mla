### Setup on remote machine:
```
> mkdir assignment2
> cd assignment2
> python3 -m venv venv
> source venv/bin/activate
> echo $VIRTUAL_ENV  // kontrolle
> pip install cuda-python
> pip install cupy-cuda13x
> pip install cuda-tile
```

--------------------------------------------------------


### Task2

## 1st run: 
> RuntimeError: Tile functions can only be called from tile code.

*Lösung:* Programm besteht aus 2 Teilen:

- **"Host" Code** (seriell auf CPU): Der Launcher
Grid festlegen, Matrix und output-Vektor definieren, Kernel starten


- **"Device" Code** (GPU-Kernel): 
@ct.kernel decorator marks this function as cuTile kernel

Die Zeilen aufteilen??
Daten (aktuelle Zeile) in ein Tile laden
Reduction



## 2nd run:

    ct.launch(torch.cuda.current_stream().cuda_stream,
TypeError: Grid must be a tuple

## 3nd run:
    ValueError: Input array is not on a CUDA device

Es war anscheinend wichtig, bei tensor device='cuda'anzugeben

     o_vector = torch.tensor((m,),
                        dtype=torch.float16,
                       device='cuda')

## 4rd run:

    cuda.tile._exception.TileTypeError: bid(): missing a required argument: 'axis'

```bid()```

 -> mögliche Werte 0,1,2
Da Grids maximal 3D sein können, sind nur diese drei Achsen möglich.

Der Rückgabewert:  der Index des aktuellen Blocks in dieser Dimension ``. Wenn der Grid zum Beispiel (20, 1, 1) groß ist, dann liefert ct.bid(0) für den ersten Block den Wert 0, für den zweiten 1, und so weiter bis 19

 ## 5th run

    cuda.tile._exception.TileTypeError: Index size 1 does not match the array rank 2
        i_tile = ct.load(i_matrix, index=(new_index,), shape=(tile_size,))

Lösung:
i_tile = ct.load(i_matrix, index=(new_index,0), shape=(1, tile_size))

Logisch, weil i_matrix 2D Matrix ist.

## 6th run

    cuda.tile._exception.TileTypeError: arange(): too many positional arguments
    "/home/mla05/assignment2/task2.py", line 33, col 15-37, in reduce:
        offsets = ct.arange(0, tile_size)

## 7th run

    cuda.tile._exception.TileTypeError: reduce(): missing a required argument: 'identity'
     "/home/mla05/assignment2/task2.py", line 36, col 14-58, in reduce:
        result = ct.reduce(i_tile, tile_size, lambda a,b: a+b)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identity = Initialwert einer Akkumulation

Identitätselement = Wert, der das Ergebnis nicht verändert. wenn er mit einem anderen Wert kombiniert wird

z.B Addition 0 weil x + 0 = x

## 8th run

    cuda.tile._exception.TileTypeError: Axis 4 is out of range for rank 2'
    "/home/mla05/assignment2/task2.py", line 36, col 14-72, in reduce:
        result = ct.reduce(i_tile, tile_size, lambda a,b: a+b, identity=0.0)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

### Task 3

## 1st run

cuda.tile._exception.TileTypeError: Invalid argument "shape" of load(): Expected a constant integer tuple, but given value is not constant
  "/home/mla05/assignment2/task3.py", line 31, col 14-75, in sum_kl:
        a_tile = ct.load(a_tensor, index = (pid_m, pid_n, 0, 0), shape = (k,l))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reason:
ct.load() (and ct.store()) require the shape parameter to be a compile-time constant — cuTile resolves tile shapes at compile time for register allocation and code generation. Runtime values like k and l (even if they don't change) are not considered constants by the compiler.

Fix:
def sum_kl(a_tensor, b_tensor, c_tensor,
           n: ct.Constant[int],
           k: ct.Constant[int],   # <-- mark as Constant

## 2nd run

cuda.tile._exception.TileTypeError: Invalid argument "shape" of load(): Expected shape length to be 4, got 2
  "/home/mla05/assignment2/task3.py", line 35, col 14-75, in sum_kl:
        a_tile = ct.load(a_tensor, index = (pid_m, pid_n, 0, 0), shape = (k,l))

The shape in ct.load() must match the full number of dimensions of the tensor. Since your tensor is 4D (M, N, K, L), shape must also have 4 elements — it describes the tile size along each dimension.


## 3rd run

cuda.tile._exception.TileTypeError: store(): multiple values for argument 'index'
  "/home/mla05/assignment2/task3.py", line 42, col 5-80, in sum_kl:
        ct.store(c_tensor, c_tile,  index = (pid_m, pid_n, 0, 0), shape = (1,1,k,l))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ct.store() has a different signature than ct.load() — the tensor being stored to is likely the third positional argument