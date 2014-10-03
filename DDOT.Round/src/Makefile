SRCS = main.cpp common.cpp DDOT.cpp DDOT.Launcher.cpp
EXE_NAME = main.out

SOURCES = $(abspath $(SRCS))

OBJECTS := ${SOURCES:.c=.o}
OBJECTS := ${OBJECTS:.cpp=.o}

HEADERS = 
INCLUDE = -I$(PATH_TO_AMD_INC) -I$(PATH_TO_NVIDIA_INC) 
COMPILERFLAGS = -Wall -W -O2 -march=native -fabi-version=0 -DGPU_PROFILING=1
CXX = g++

is_64 = $(shell s=`uname -m`; if (echo $$s | grep x86_64 > /dev/null); then echo 1; fi)

#CXXFLAGS = $(COMPILERFLAGS) $(DEFINES:%=-D%) $(INCLUDE)
CXXFLAGSAMD = $(COMPILERFLAGS) -masm=att -DAMD $(DEFINES:%=-D%) $(INCLUDE)
CXXFLAGSNVIDIA = $(COMPILERFLAGS) -masm=intel $(DEFINES:%=-D%) $(INCLUDE)

amd: CXXFLAGS = $(CXXFLAGSAMD)
nvidia: CXXFLAGS = $(CXXFLAGSNVIDIA)

all: amd nvidia

# AMD
PATH_TO_AMD_INC=/opt/sdks/amd/current/include/ -I$(MPFR_HOME)/include/ -I$(GMP_HOME)/include/
PATH_TO_AMD_LIB=/usr/lib/ -lOpenCL -L$(MPFR_HOME)/lib/ -lmpfr -L$(GMP_HOME)/lib/ -lgmp

# NVIDIA
PATH_TO_NVIDIA_INC=/usr/local/cuda/include/ -I$(MPFR_HOME)/include/ -I$(GMP_HOME)/include/
PATH_TO_NVIDIA_LIB=/usr/lib -lOpenCL -L/usr/local/cuda/lib64 -L$(MPFR_HOME)/lib -lmpfr -L$(GMP_HOME)/lib -lgmp
#for ics machine
PATH_TO_NVIDIA_LIB+=-L$(MPFR_HOME)/lib64/ -L$(GMP_HOME)/lib64/

amd:    $(join $(EXE_NAME),.amd)
nvidia: $(join $(EXE_NAME),.nvidia)

%.amd: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_AMD_INC) -Wl,-rpath,$(PATH_TO_AMD_LIB)  -o $@ $(OBJECTS) -L$(PATH_TO_AMD_LIB)

%.nvidia: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_NVIDIA_INC) -Wl,-rpath,$(PATH_TO_NVIDIA_LIB) -o $@ $(OBJECTS) -L$(PATH_TO_NVIDIA_LIB)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_AMD_INC) -I$(PATH_TO_AMD_INC) -Wl,-rpath,$(PATH_TO_AMD_LIB):$(PATH_TO_AMD_LIB) -c $< -L$(PATH_TO_AMD_LIB) -L$(PATH_TO_AMD_LIB)

clean:
	rm -f main.out.amd main.out.nvidia *.o
