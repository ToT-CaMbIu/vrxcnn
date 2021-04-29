exe_name = main

h5_part = -I/usr/local/opt/szip/include\
-L/usr/local/Cellar/hdf5/1.12.0_3/lib \
/usr/local/Cellar/hdf5/1.12.0_3/lib/libhdf5_hl_cpp.a \
/usr/local/Cellar/hdf5/1.12.0_3/lib/libhdf5_cpp.a \
/usr/local/Cellar/hdf5/1.12.0_3/lib/libhdf5_hl.a \
/usr/local/Cellar/hdf5/1.12.0_3/lib/libhdf5.a \
-L/usr/local/opt/szip/lib -lsz -lz -ldl -lm

LAYERS_DIR = ./hosts
SRCS = utils.cpp main.cpp opencl_environment.cpp $(LAYERS_DIR)/*.cpp

all: cpp

flags = -std=c++17 -O2

cpp: $(SRCS)
	g++ $(flags) $^ -o $(exe_name) -framework OpenCL

h5: $(SRCS)
	g++ $(h5_part) $(flags) $^ h5_helper.cpp -o $(exe_name) -framework OpenCL

clean:
	rm -rf $(exe_name) *.o .depend

clean-all: clean
	rm -rf *.pocl *.dump

#vortex region

LLVM_PREFIX ?= /opt/llvm-riscv
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf
POCL_CC_PATH ?= /opt/pocl/compiler
POCL_RT_PATH ?= /opt/pocl/runtime

VORTEX_DRV_PATH ?= $(realpath ../../../driver)
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

K_LLCFLAGS += "-O3 -march=riscv32 -target-abi=ilp32f -mcpu=generic-rv32 -mattr=+m,+f -float-abi=hard -code-model=small"
K_CFLAGS   += "-v -O3 --sysroot=$(SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -march=rv32imf -mabi=ilp32f -I$(VORTEX_RT_PATH)/include -fno-rtti -fno-exceptions -ffreestanding -nostartfiles -fdata-sections -ffunction-sections"
K_LDFLAGS  += "-Wl,-Bstatic,-T$(VORTEX_RT_PATH)/linker/vx_link.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a -lm"

CXXFLAGS += -std=c++17 -O2 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(POCL_RT_PATH)/include -I.

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_DRV_PATH)/simx -lOpenCL -lvortex

KERNEL_NAME += kernel_matrix_mul

PROJECT = matrix_mul_test

vortex: $(PROJECT) $(KERNEL_NAME).pocl

$(KERNEL_NAME).pocl: $(KERNEL_NAME).cl
	LLVM_PREFIX=$(LLVM_PREFIX) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_PREFIX)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -LLCFLAGS $(K_LLCFLAGS) -CFLAGS $(K_CFLAGS) -LDFLAGS $(K_LDFLAGS) -o $(KERNEL_NAME).pocl $(KERNEL_NAME).cl

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) $(KERNEL_NAME).pocl
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT)

