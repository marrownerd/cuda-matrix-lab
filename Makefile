# Компилятор (NVIDIA CUDA Compiler)
NVCC = nvcc

# Флаги компиляции
# -O3      : Включить оптимизацию кода
# -lcublas : Подключить библиотеку cuBLAS
NVCC_FLAGS = -O3 -lcublas

# Имя исполняемого файла, который мы хотим получить
TARGET = matrix_mul

# Исходный файл с кодом
SRC = compare_methods.cu

# Цель по умолчанию (выполняется при наборе 'make')
all: $(TARGET)

# Правило сборки: как из .cu сделать исполняемый файл
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Правило очистки: удалить скомпилированный файл (команда 'make clean')
clean:
	rm -f $(TARGET) result.txt