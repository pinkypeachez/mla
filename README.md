## Vorgeschichte

Graphics blabla 


## Data Flow

**Input:** Data kommt aus CPU-RAM.

**Transfer:** Daten werden über den PCIe-Bus in den VRAM (entweder GDDR oder HBM) kopiert

**Verarbeitung:** GPU-Kerne benutzen Shared Memory (L1/L2 Caches für Zwischenergebnisse)

**Output:** das ferige Ergebnis wird im GPU DRAM - VRAM abgelegt

*VRAM [Video Random Access Memory] Typen:*

*-- **GDDR** [Graphics Double Data Rate]: Speicherchips sind auf dem Board um den Grafikchip herum platziert*

*-- **HBM** [High Bandwidth Memory]: der Speicher wird in Schichten direkt auf dem GPU-Die dem Silizium-Chip gestapelt, "on-package" - im selben Gehäuse, aber "off-chip" weil nicht IM eigentlichen Rechenkern*

**Rücktransfer:** VRAM -> System-RAM

## Memory-Hierarchie
**Die GPU**
Die gesamte Grafikkarte besitzt den VRAM (also GDDR oder HBM) + L2-Cache (auf den alle Teile der GPU zugreifen können)

**Der Streaming Multiprocessor (SM)**
besitzt eigenen L1 Cache + Shared Memory.

**CUDA-Core/ "Kern"**
ALU + Register

## GPU in Detail
GPU (Durchsatz optimiert!)
Ziel: so viele Datenpunkte wie möglich gleichzeitig zu verarbeiten
Grob gesagt: sehr viele kleine ALUs

Ausführungsmodell: SIMT [Single Instruction, Multiple Threads]
1 Befehl -> auf viele Daten gleichzeitig angewendet

Andere Hersteller:
| Hersteller | GPU-Serie | Software-Stack (Gegenstück zu CUDA) | Programmiersprache / API |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | GeForce, RTX, A100, H100 | **CUDA** | CUDA C++ / Python |
| **AMD** | Radeon, Instinct (MI-Serie) | **ROCm** (Radeon Open Compute) | HIP / C++ |
| **Intel** | Arc, Data Center Max, Gaudi | **oneAPI** | Data Parallel C++ (DPC++) / SYCL |
| **Apple** | M-Serie (M1, M2, M3, M4) | **Metal** | Metal Shading Language / C++ |

Ein bisschen Geopolitik xD:
Google hat das Framework TensorFlow entwickelt.
Google hat eigene Hardware ("Alternative" zu NVIDIA GPU) entwickelt -  TPU [Tensor Processing Units]

TensorFlow (Google)
PyTorch (Meta/ Facebook)



CUDA = NVIDIAs Programmierplattform für General-Purpose GPU Computing
Idee: Entwicklern zu ermöglichen, die Rechenleistung der GPU für allgemeine mathematische Berechnungen zu nutzen (GPGPU – General Purpose Computation on Graphics Processing Units), anstatt sie nur für Grafik-Rendering zu verwenden

Begriffe:
"Host": CPU
"Device": GPU
Programm läuft auf CPU, rechenintensive Teile erledigt GPU

"Kernel": Funktion, die von CPU aufgerufen wird um xxxx-fach parallel auf der GPU ausgeführt zu werden



**Hierarchie [von unten nach oben]:**
Ebene 1: CUDA Core
Ein Rechenkern, 1 FP Operation/ Takt

Ebene 2: Warp [32 Threads] = atomare Ausführungseinheit der GPU (bei AMD heißt es "Wavefront")

Alle 32 Threads in einem Warp bekommen exakt denselben Befehl zur exakt selben Zeit. Aber sie führen den auf unterschiedlichen Daten aus

Ebene 3 — Streaming Multiprocessor (SM)