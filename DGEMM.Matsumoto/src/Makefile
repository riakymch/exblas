# This Makefile compiles the CBLAS routines
#
OCLBLASROOT	= ../.
include $(OCLBLASROOT)/Makefile.in

dummy: all

#
# Erase all object and archive files
#
clean:
	$(RM) *.o a.out core 

# Error handling routines for level 2 & 3

errhand = oclblas_xerbla.o

# Object files of all routines
alev = $(alev1) $(alev2) $(alev3) $(errhand) 
#
#
# CBLAS routines
#
# Level 1
#
# 

#
# All object files for single real precision
#
slev1 =
#
# All object files for double real precision
#
dlev1 = oclblas_dcopy.o
#
# All object files for single complex precision
#
clev1 =
#
# All object files for double complex precision
#
zlev1 =
#
# Common files for single / complex precision
#
sclev1 =
#
# All object files
#
alev1 = $(slev1) $(dlev1) $(clev1) $(zlev1) $(sclev1)


#
# Make an archive file
#

# Single real precision
slib1: $(slev1) $(sclev1)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(slev1) $(sclev1)
	$(RANLIB) $(OCLBLASLIB)

# Double real precision
dlib1: $(dlev1) 
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(dlev1)
	$(RANLIB) $(OCLBLASLIB)

# Single complex precision
clib1: $(clev1) $(sclev1)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(clev1) $(sclev1)
	$(RANLIB) $(OCLBLASLIB)

# Double complex precision
zlib1: $(zlev1)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(zlev1)
	$(RANLIB) $(OCLBLASLIB)

# All precisions
all1:  $(alev1)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(alev1)
	$(RANLIB) $(OCLBLASLIB)

#
#
# CBLAS routines
#
# Level 2
#
# 

#
# All object files for single real precision
#
slev2 =
#
# All object files for double real precision
#
dlev2 =
#
# All object files for single complex precision
#
clev2 =
#
# All object files for double complex precision
#
zlev2 =
#
# All object files
#
alev2 = $(slev2) $(dlev2) $(clev2) $(zlev2)

#
# Make an archive file
#

# Single real precision
slib2: $(slev2) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(slev2) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# Double real  precision
dlib2: $(dlev2) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(dlev2) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# Single complex precision
clib2: $(clev2) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(clev2) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# Double complex precision
zlib2: $(zlev2) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(zlev2) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# All precisions
all2:  $(alev2) $(errhand)
	$(ARCH) $(ARCHFLAGS)  $(OCLBLASLIB) $(alev2) $(errhand)
	$(RANLIB) $(OCLBLASLIB)
#
#
# CBLAS routines
#
# Level 3
#
# 

#
# All object files for single real precision
#
slev3 = oclblas_sgemm.o
	#oclblas_ssymm.o oclblas_ssyrk.o oclblas_ssyr2k.o \
	#oclblas_strmm.o
#
# All object files for double real precision
#
dlev3 = oclblas_dgemm.o 
	#oclblas_dsymm.o oclblas_dsyrk.o oclblas_dsyr2k.o \
	#oclblas_dtrmm.o oclblas_dtrsm.o

#
# All object files for single complex precision
#
clev3 =
#
# All object files for double complex precision
#
zlev3 =
#
# All object files
#
alev3 = $(slev3) $(dlev3) $(clev3) $(zlev3)

#
# Make an archive file
#

# Single real precision
slib3: $(slev3) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(slev3) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# Double real precision
dlib3: $(dlev3) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(dlev3) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# Single complex precision
clib3: $(clev3) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(clev3) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# Single complex precision
zlib3: $(zlev3) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(zlev3) $(errhand)
	$(RANLIB) $(OCLBLASLIB)

# All precisions
all3:  $(alev3) $(errhand)
	$(ARCH) $(ARCHFLAGS) $(OCLBLASLIB) $(alev3) 
	$(RANLIB) $(OCLBLASLIB)

# All levels and precisions
all: $(alev)  
	$(ARCH) $(ARCHFLAGS)  $(OCLBLASLIB) $(alev) 
	$(RANLIB) $(OCLBLASLIB)


.SUFFIXES: .o .c .cpp .f

.c.o:
	$(CC) $(CFLAGS) -I../include -I../interface -I. -c $*.c
.cpp.o:
	$(CXX) $(CFLAGS) -I../include -I../interface -I. -c $*.cpp
.f.o:
	$(FC) $(FFLAGS) -c $*.f
