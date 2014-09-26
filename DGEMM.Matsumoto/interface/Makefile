OCLBLASROOT = ../.
include $(OCLBLASROOT)/Makefile.in

ALLSRC = \
	 oclblas_interface.cpp \
	 oclblas_internal_interface.cpp \
	 oclblas_runtime.cpp

ALLOBJ = $(ALLSRC:.cpp=.o)


all: $(OCLBLASLIBNAME)

$(OCLBLASLIBNAME): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(ALLOBJ)
	$(RANLIB) $(OCLBLASLIB)

.cpp.o:
	$(CXX) $(OPTS) $(CXXFLAGS) -c $<

clean:
	$(RM) $(OCLBLASLIB) $(ALLOBJ) 

cleanall: clean
	$(RM) *.o *.a *~
