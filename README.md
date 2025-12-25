# Умножение матриц на CUDA

Сравнение производительности различных методов умножения матриц ($C = A \times B$):
1. **CPU** (последовательный алгоритм)
2. **Naive GPU** (глобальная память)
3. **Shared Memory GPU** (блочный алгоритм с разделяемой памятью)
4. **cuBLAS** (библиотека NVIDIA)


Загрузить модуль CUDA:
module load cuda (или module load nvidia/cuda — зависит от кластера, можно проверить через module avail).

code Bash

    
nvcc compare_methods.cu -o benchmark -lcublas -O3


Флаг -O3, чтобы CPU не тормозил слишком сильно.

code Bash

    
./benchmark

  
