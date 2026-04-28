# Parallel Game of Life -- build system
#
# Targets:
#   gol_serial   -- single-node CPU reference, no MPI / no CUDA
#   gol_mpi_cpu  -- MPI build, CPU compute step
#   gol_mpi_cuda -- MPI build, CUDA compute step (CUDA-aware MPI)
#   diff_grids   -- raw-grid comparison tool
#   all          -- everything
#
# AiMOS DCS modules expected in the environment:
#   module load xl_r spectrum-mpi cuda/11.2
#
# Override toolchain via env: CC, MPICC, NVCC, CUDA_HOME.

CC       ?= gcc
MPICC    ?= mpicc
NVCC     ?= nvcc
CUDA_HOME ?= /usr/local/cuda

CFLAGS   ?= -O3 -std=c99 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -Iinclude
NVFLAGS  ?= -O3 -std=c++11 -Iinclude -gencode arch=compute_70,code=sm_70 \
            -Xcompiler "-O3 -Wall"
LDFLAGS  ?=
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64 -lcudart

BIN := bin
SRC := src

.PHONY: all clean dirs
all: dirs $(BIN)/gol_serial $(BIN)/gol_mpi_cpu $(BIN)/gol_mpi_cuda $(BIN)/diff_grids

dirs:
	@mkdir -p $(BIN)

# Shared objects (no USE_CUDA branching inside these files)
$(BIN)/gol_mpi.o:      $(SRC)/gol_mpi.c      | dirs
	$(MPICC) $(CFLAGS) -c -o $@ $<
$(BIN)/gol_io.o:       $(SRC)/gol_io.c       | dirs
	$(MPICC) $(CFLAGS) -c -o $@ $<
$(BIN)/gol_cpu_step.o: $(SRC)/gol_cpu_step.c | dirs
	$(CC)    $(CFLAGS) -c -o $@ $<

# Serial reference
$(BIN)/gol_serial: $(SRC)/gol_serial.c include/clockcycle.h | dirs
	$(CC) $(CFLAGS) -o $@ $<

# MPI / CPU
$(BIN)/main_cpu.o: $(SRC)/main.c | dirs
	$(MPICC) $(CFLAGS) -c -o $@ $<

$(BIN)/gol_mpi_cpu: $(BIN)/main_cpu.o $(BIN)/gol_mpi.o $(BIN)/gol_io.o $(BIN)/gol_cpu_step.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# MPI / CUDA
$(BIN)/main_gpu.o: $(SRC)/main.c | dirs
	$(MPICC) $(CFLAGS) -DUSE_CUDA -I$(CUDA_HOME)/include -c -o $@ $<
$(BIN)/gol_cuda.o: $(SRC)/gol_cuda.cu | dirs
	$(NVCC) $(NVFLAGS) -DUSE_CUDA -c -o $@ $<

$(BIN)/gol_mpi_cuda: $(BIN)/main_gpu.o $(BIN)/gol_mpi.o $(BIN)/gol_io.o \
                     $(BIN)/gol_cpu_step.o $(BIN)/gol_cuda.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(CUDA_LDFLAGS) $(LDFLAGS)

# Diff tool
$(BIN)/diff_grids: tools/diff_grids.c | dirs
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(BIN)
