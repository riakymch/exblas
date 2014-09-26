include ../Makefile.in

LIBNAME	= libauxiliary.a
ALLOBJ	= benchmark.o s_benchmark.o d_benchmark.o

.PHONY: all clean test

all: $(LIBNAME)

$(LIBNAME): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) $(LIBNAME) $(ALLOBJ)
