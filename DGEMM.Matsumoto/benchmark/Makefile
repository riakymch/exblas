all:
	$(MAKE) all -C auxiliary
	$(MAKE) all -C dcopy
	$(MAKE) all -C dgemm
	#$(MAKE) all -C dsymm
	#$(MAKE) all -C dsyrk
	#$(MAKE) all -C dsyr2k
	#$(MAKE) all -C dtrmm
	#$(MAKE) all -C dtrsm
	$(MAKE) all -C sgemm
	#$(MAKE) all -C ssymm
	#$(MAKE) all -C ssyrk
	#$(MAKE) all -C ssyr2k
	#$(MAKE) all -C strmm

clean:
	$(MAKE) clean -C auxiliary
	$(MAKE) clean -C dcopy
	$(MAKE) clean -C dgemm
	#$(MAKE) clean -C dsymm
	#$(MAKE) clean -C dsyrk
	#$(MAKE) clean -C dsyr2k
	#$(MAKE) clean -C dtrmm
	#$(MAKE) clean -C dtrsm
	$(MAKE) clean -C sgemm
	#$(MAKE) clean -C ssymm
	#$(MAKE) clean -C ssyrk
	#$(MAKE) clean -C ssyr2k
	#$(MAKE) clean -C strmm

test:
	$(MAKE) all -C auxiliary
	#$(MAKE) test -C dcopy
	$(MAKE) test -C dgemm
	#$(MAKE) test -C dsymm
	#$(MAKE) test -C dsyrk
	#$(MAKE) test -C dsyr2k
	#$(MAKE) test -C dtrmm
	#$(MAKE) test -C dtrsm
	$(MAKE) test -C sgemm
	#$(MAKE) test -C ssymm
	#$(MAKE) test -C ssyrk
	#$(MAKE) test -C ssyr2k
	#$(MAKE) test -C strmm
