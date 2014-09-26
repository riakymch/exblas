OCLBLASROOT = ../.
include $(OCLBLASROOT)/Makefile.in

TARGET = cl2bin

ALLSRC = \
	 cl2bin.cpp

ALLKERNEL = \
	 dcopy.bin \
	 dgemm.bin \
	 sgemm.bin

ALLOBJ = $(ALLSRC:.cpp=.o)

CXXFLAGS += -I../interface


all: $(TARGET) $(ALLKERNEL)

$(TARGET): $(ALLOBJ) $(OCLBLASLIB)
	$(CXX) $(ALLOBJ) $(LDFLAGS) -o $@

.cpp.o:
	$(CXX) $(OPTS) $(CXXFLAGS) -c $< -o $@

.cl.bin:
	OCLBLAS_KERNEL_DIR=./ ./cl2bin $<

clean:
	$(RM) $(TARGET) $(ALLOBJ) $(ALLKERNEL)

cleanall: clean
	$(RM) *~

.SUFFIXES:	.cl .bin
