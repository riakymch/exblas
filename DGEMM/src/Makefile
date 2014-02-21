SRCS = main.cpp Kulisch.Device.Launcher.cpp Kulisch.Host.cpp common.cpp Super.Accumulator.cpp
EXE_NAME = main.out

SOURCES = $(abspath $(SRCS))

OBJECTS := ${SOURCES:.c=.o}
OBJECTS := ${OBJECTS:.cpp=.o}

HEADERS = 
INCLUDE = -I$(PATH_TO_NVIDIA_INC) -I$(PATH_TO_APPLE_INC) 
COMPILERFLAGS = -c -Wall -g -O2 -Wshorten-64-to-32
CXX = g++

all: amd apple intel nvidia

is_64=$(shell s=`uname -m`; if (echo $$s | grep x86_64 > /dev/null); then echo 1; fi)

CXXFLAGS=-Wall -W -O2 $(DEFINES:%=-D%) $(INCLUDE) -march=native -fabi-version=0 -fopenmp -masm=intel -DGPU_PROFILING=1

# AMD
PATH_TO_AMD_INC=/opt/AMDAPP/include/
ifeq ($(is_64), 1)
PATH_TO_AMD_LIB=/opt/AMDAPP/lib/x86_64/
else
PATH_TO_AMD_LIB=/opt/AMDAPP/lib/x86/
endif

# Intel
PATH_TO_INTEL_INC=/usr/include/
ifeq ($(is_64), 1)
PATH_TO_INTEL_LIB=/usr/lib64/
endif

# NVIDIA
PATH_TO_NVIDIA_INC=/usr/local/cuda/include/ -I$(HOME)/soft/lib/mpfr/include/ -I$(HOME)/soft/lib/gmp/include/
PATH_TO_NVIDIA_LIB=/usr/lib/ -L$(HOME)/soft/lib/mpfr/lib/ -I$(HOME)/soft/lib/gmp/lib/ -lmpfr -lgmp -lOpenCL

# MAC
PATH_TO_APPLE_INC=/opt/local/include/
PATH_TO_APPLE_LIB=/opt/local/lib/

amd:    $(join $(EXE_NAME),.amd)
apple:  $(join $(EXE_NAME),.apple)
intel:  $(join $(EXE_NAME),.intel)
nvidia: $(join $(EXE_NAME),.nvidia)

%.amd:$(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_AMD_INC) -L$(PATH_TO_AMD_LIB) -Wl,-rpath,$(PATH_TO_AMD_LIB)  -o $@ $(OBJECTS) -lOpenCL

%.apple:$(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_APPLE_INC) -L$(PATH_TO_APPLE_LIB) -Wl,-rpath,$(PATH_TO_APPLE_LIB) -framework opencl -o $@ $(OBJECTS)

%.intel:$(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_INTEL_INC) -L$(PATH_TO_INTEL_LIB) -Wl,-rpath,$(PATH_TO_INTEL_LIB) -o $@ $(OBJECTS) -lOpenCL

%.nvidia:$(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_NVIDIA_INC) -Wl,-rpath,$(PATH_TO_NVIDIA_LIB) -o $@ $(OBJECTS) -L$(PATH_TO_NVIDIA_LIB)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_NVIDIA_INC) -Wl,-rpath,$(PATH_TO_NVIDIA_LIB) -c $< -L$(PATH_TO_NVIDIA_LIB)

clean:
	rm -f main.out.amd *.apple *.intel main.out.nvidia *.o
