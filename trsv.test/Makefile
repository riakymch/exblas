SRCS = main.cpp common.cpp
EXE_NAME = main.out

SOURCES = $(abspath $(SRCS))

OBJECTS := ${SOURCES:.c=.o}
OBJECTS := ${OBJECTS:.cpp=.o}

all: main.out

is_64=$(shell s=`uname -m`; if (echo $$s | grep x86_64 > /dev/null); then echo 1; fi)

CXX = g++
CXXFLAGS=-Wall -W -O3 -g $(DEFINES:%=-D%) -march=native -fabi-version=0 -fopenmp -masm=intel

PATH_TO_INC=$(HOME)/soft/lib/mpfr/include/ -I$(HOME)/soft/lib/gmp/include/
PATH_TO_LIB=$(HOME)/soft/lib/mpfr/lib/ -L$(HOME)/soft/lib/gmp/lib/ -lmpfr -lgmp

main.out:$(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_INC) -Wl,-rpath,$(PATH_TO_LIB)  -o $@ $(OBJECTS) -L$(PATH_TO_LIB)
	
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_INC) -Wl,-rpath,$(PATH_TO_LIB) -c $< -L$(PATH_TO_LIB)
	
%.o : %.c
	$(CXX) $(CXXFLAGS) -I$(PATH_TO_INC) -Wl,-rpath,$(PATH_TO_LIB) -c $< -L$(PATH_TO_LIB)

clean:
	rm -f main.out *.o
