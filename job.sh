#!/bin/bash
#SBATCH --job-name=matrix_cuda      # Имя задачи
#SBATCH --output=result.txt         # Куда писать вывод
#SBATCH --error=error.txt           # Куда писать ошибки
#SBATCH --ntasks=1                  # 1 задача
#SBATCH --cpus-per-task=1           # 1 CPU
#SBATCH --gpus=1                    # 1 GPU (ОБЯЗАТЕЛЬНО!)
#SBATCH --time=00:05:00             # Лимит времени 5 минут

# Подгрузка модулей (раскомментируй нужное, зависит от кластера)
module load cuda
# module load nvidia/cuda

# Запуск программы
./matrix_mul