OCLBLASROOT	= ./.
include $(OCLBLASROOT)/Makefile.in

all: $(OCLBLASLIBNAME)


# In general, the Makefile call other Makefiles in the sub-directories.


clean:
	( cd clkernel && make clean )
	( cd interface && make clean )
	( cd src && make clean )

cleanobj:
	( cd src && make clean )

rmlib:
	( rm -f $(OCLBLASROOT)/lib/$(OCLBLASLIBNAME) )

$(OCLBLASLIBNAME):
	( cd src && make all)
	( cd interface && make all )
	( cd clkernel && make all )

test:
	( cd benchmark && make clean test )
install:
	#( mkdir -p /opt/beta/oclblas-$(VERSION)/clkernel )
	( mkdir -p ./beta/oclblas-$(VERSION)/clkernel )
	( cp -fr include lib ./beta/oclblas-$(VERSION) )
	( cp -fr clkernel/*.bin ./beta/oclblas-$(VERSION)/clkernel )
