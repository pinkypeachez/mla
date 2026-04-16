### Setup on remote machine:
> mkdir assignment2
> cd assignment2
> python3 -m venv venv
> source venv/bin/activate
> echo $VIRTUAL_ENV  // kontrolle
> pip install cuda-python
> pip install cupy-cuda13x
> pip install cuda-tile

--------------------------------------------------------


### Task2

1st run: 
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

bid()
 - mögliche Werte 0,1,2
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