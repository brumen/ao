# to compile xmm files
SWIG = swig
PROC_DIR = -funroll-all-loops -falign-functions=16 -O3
PROC_DIR_AVX = -funroll-all-loops -falign-functions=32 -O3
XMM = $(PROC_DIR) -msse4a
AVX2 = $(PROC_DIR_AVX) -mavx2
PYTHON_INCLUDE = $(shell python3.7-config --includes) $(shell python3.5-config --includes) -I/usr/local/lib/python3.7/dist-packages/numpy/core/include/
GCC = g++
OMP = -fopenmp

vtpm_cpu_avx: vtpm_cpu.c vtpm_cpu.i
	$(SWIG) -python -o cxx/vtpm_cpu_wrap3.c -I. -Wall vtpm_cpu3.i
	$(GCC) $(OMP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -c vtpm_cpu.c -o cxx/vtpm_cpu3.o
	$(GCC) $(OMP) $(AVX2) $(PYTHON_INCLUDE) -fPIC -S vtpm_cpu.c -o cxx/vtpm_cpu3.s
	$(GCC) $(OMP) $(AVX2) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/vtpm_cpu_wrap3.c -o cxx/vtpm_cpu_wrap3.o
	$(GCC) -shared cxx/vtpm_cpu3.o cxx/vtpm_cpu_wrap3.o -lgomp -o _vtpm_cpu3.so

vtpm_cpu_xmm: vtpm_cpu.c vtpm_cpu.i
	$(SWIG) -python -o cxx/vtpm_cpu_wrap.c -I. -Wall vtpm_cpu3.i
	$(GCC) $(OMP) $(XMM) $(PYTHON_INCLUDE) -fPIC -c vtpm_cpu.c -o cxx/vtpm_cpu3.o
	$(GCC) $(OMP) $(XMM) $(PYTHON_INCLUDE) -fPIC -S vtpm_cpu.c -o cxx/vtpm_cpu3.s
	$(GCC) $(OMP) $(XMM) $(PYTHON_INCLUDE) -I. -fPIC -c cxx/vtpm_cpu_wrap.c -o cxx/vtpm_cpu_wrap3.o
	$(GCC) -shared cxx/vtpm_cpu3.o cxx/vtpm_cpu_wrap3.o -lgomp -o _vtpm_cpu3.so
