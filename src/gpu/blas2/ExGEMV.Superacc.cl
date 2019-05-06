
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable  //For 64 atomic operations
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
    #pragma OPENCL EXTENSION cl_khr_fp64               : enable  //For double precision numbers
    #pragma OPENCL EXTENSION cl_nv_pragma_unroll       : enable
#endif

#define BIN_COUNT  39
#define K           8                   //High-radix carry-save bits
#define digits     56
#define deltaScale 72057594037927936.0  //Assumes K > 0
#define f_words    20
#define TSAFE       0

#define ROW_DIM 0
#define COL_DIM 1


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions
////////////////////////////////////////////////////////////////////////////////
double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

// signedcarry in {-1, 0, 1}
long xadd(__global volatile long *sa, long x, uchar *of) {
    // OF and SF  -> carry=1
    // OF and !SF -> carry=-1
    // !OF        -> carry=0
    long y = atom_add(sa, x);
    long z = y + x; // since the value sa->accumulator[i] can be changed by another work item

    // TODO: cover also underflow
    *of = 0;
    if(x > 0 && y > 0 && z < 0)
        *of = 1;
    if(x < 0 && y < 0 && z > 0)
        *of = 1;

    return y;
}


////////////////////////////////////////////////////////////////////////////////
// Kulisch accumulator: rounding and accumulation functions
////////////////////////////////////////////////////////////////////////////////
double OddRoundSumNonnegative(double th, double tl) {
    union {
        double d;
        long l;
    } thdb;

    thdb.d = th + tl;
    // - if the mantissa of th is odd, there is nothing to do
    // - otherwise, round up as both tl and th are positive
    // in both cases, this means setting the msb to 1 when tl>0
    thdb.l |= (tl != 0.0);
    return thdb.d;
}

int Normalize(__global long *accumulator, int *imin, int *imax) {
    long carry_in = accumulator[*imin] >> digits;
    accumulator[*imin] -= carry_in << digits;
    int i;
    // Sign-extend all the way
    for (i = *imin + 1; i < BIN_COUNT; ++i) {
        accumulator[i] += carry_in;
        long carry_out = accumulator[i] >> digits;    // Arithmetic shift
        accumulator[i] -= (carry_out << digits);
        carry_in = carry_out;
    }
    *imax = i - 1;

    // Do not cancel the last carry to avoid losing information
    accumulator[*imax] += carry_in << digits;

    return carry_in < 0;
}

double Round(__global long *accumulator) {
    int imin = 0;
    int imax = 38;
    int negative = Normalize(accumulator, &imin, &imax);

    //Find leading word
    int i;
    //Skip zeroes
    for (i = imax; accumulator[i] == 0 && i >= imin; --i) {
    }
    if (negative) {
        //Skip ones
        for (; (accumulator[i] & ((1l << digits) - 1)) == ((1l << digits) - 1) && i >= imin; --i) {
        }
    }
    if (i < 0)
        //TODO: should we preserve sign of zero?
        return 0.0;

    long hiword = negative ? ((1l << digits) - 1) - accumulator[i] : accumulator[i];
    double rounded = (double) hiword;
    double hi = ldexp(rounded, (i - f_words) * digits);
    if (i == 0)
        return negative ? -hi : hi;  // Correct rounding achieved
    hiword -= (long) rint(rounded);
    double mid = ldexp((double) hiword, (i - f_words) * digits);

    //Compute sticky
    long sticky = 0;
    for (int j = imin; j != i - 1; ++j)
        sticky |= negative ? (1l << digits) - accumulator[j] : accumulator[j];

    long loword = negative ? (1l << digits) - accumulator[(i - 1)] : accumulator[(i - 1)];
    loword |= !!sticky;
    double lo = ldexp((double) loword, (i - 1 - f_words) * digits);

    //Now add3(hi, mid, lo)
    //No overlap, we have already normalized
    if (mid != 0)
        lo = OddRoundSumNonnegative(mid, lo);

    //Final rounding
    hi = hi + lo;
    return negative ? -hi : hi;
}

void AccumulateWord(__global volatile long *sa, int i, long x) {
    // With atomic accumulator updates
    // accumulation and carry propagation can happen in any order,
    // as long as addition is atomic
    // only constraint is: never forget an overflow bit
    uchar overflow;
    long carry = x;
    long carrybit;
    long oldword = xadd(&sa[i], x, &overflow);

    // To propagate over- or underflow
    while (overflow) {
        // Carry or borrow
        // oldword has sign S
        // x has sign S
        // accumulator[i] has sign !S (just after update)
        // carry has sign !S
        // carrybit has sign S
        carry = (oldword + carry) >> digits;    // Arithmetic shift
        bool s = oldword > 0;
        carrybit = (s ? 1l << K : -1l << K);

        // Cancel carry-save bits
        xadd(&sa[i], (long) -(carry << digits), &overflow);
        if (TSAFE && (s ^ overflow))
            carrybit *= 2;
        carry += carrybit;

        ++i;
        if (i >= BIN_COUNT)
            return;
        oldword = xadd(&sa[i], carry, &overflow);
    }
}

void Accumulate(__global volatile long *sa, double x) {
    if (x == 0)
        return;

    int e;
    frexp(x, &e);
    int exp_word = e / digits;  // Word containing MSbit
    int iup = exp_word + f_words;

    double xscaled = ldexp(x, -digits * exp_word);

    int i;
    for (i = iup; xscaled != 0; --i) {
        double xrounded = rint(xscaled);
        long xint = (long) xrounded;

        AccumulateWord(sa, i, xint);

        xscaled -= xrounded;
        xscaled *= deltaScale;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Matrix-vector multiplication algorithm
////////////////////////////////////////////////////////////////////////////////
__kernel void gemv(
    const uint m,
    const uint n,
    const double alpha,
    __global double *a,
    const uint lda,
    const uint offseta,
    __global double *x,
    const uint incx,
    const uint offsetx,
    const double beta,
    __global double *y,
    const uint incy,
    const uint offsety,
    __local double *work,
    __global long *d_Superaccs
){
    // Load a slice of X in WORK, using all available threads
    int ncols = n / get_global_size(COL_DIM); // nb values to load
    int col0 = ncols * get_global_id(COL_DIM); // first value to load
	if ((offsetx == 0) && (incx == 1)) {  
		for (int k = 0; k < ncols; k += get_local_size(ROW_DIM)) {
			int col = k + get_local_id(ROW_DIM);
			if (col < ncols)
				work[col] = x[col0 + col];
		}
	} else {
		for (int k = 0; k < ncols; k += get_local_size(ROW_DIM)) {
			int col = k + get_local_id(ROW_DIM);
			if (col < ncols)
				work[col] = x[offsetx + incx * (col0 + col)];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); // sync group

    __global long *l_working = d_Superaccs + (get_global_id(ROW_DIM) + m * get_global_id(COL_DIM))* BIN_COUNT;
    // Initialize accumulators
    for (uint i = 0; i < BIN_COUNT; i++) {
        l_working[i] = 0.0;
    }

    // Compute partial dot product
    if (get_global_id(ROW_DIM) < m) {
		if (offseta == 0) {  
			for (int k = 0; k < ncols; k++) {
                double xs, r;
			    xs = TwoProductFMA(a[get_global_id(ROW_DIM) + lda * (col0 + k)], alpha * work[k], &r);

			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
			}
		} else {
			for (int k = 0; k < ncols; k++) {
                double xs, r;
			    xs = TwoProductFMA(a[offseta + get_global_id(ROW_DIM) + lda * (col0 + k)], alpha * work[k], &r);

			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
			}
		}
    }

    // Store in Y (P columns per row)
    if (get_global_id(ROW_DIM) < m) {
        if ((offsety == 0) && (incy == 1)) {
			if (beta == 0.0) {
				y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] = Round(l_working);
			} else if (beta == 1.0) {
                Accumulate(l_working, y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)]);
				y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] = Round(l_working);
			} else {
                double xs, r;
                xs = TwoProductFMA(beta, y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)], &r);
			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
				y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] = Round(l_working);
			}
        } else {
			if (beta == 0.0) {
				y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] = Round(l_working);
			} else if (beta == 1.0) {
                Accumulate(l_working, y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))]);
				y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] = Round(l_working);
			} else {
                double xs, r;
                xs = TwoProductFMA(beta, y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))], &r);
			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
				y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] = Round(l_working);
			}
        }
    }
}


__kernel void gemvT(
    const uint m,
    const uint n,
    const double alpha,
    __global double *a,
    const uint lda,
    const uint offseta,
    __global double *x,
    const uint incx,
    const uint offsetx,
    const double beta,
    __global double *y,
    const uint incy,
    const uint offsety,
    __local double *work,
    __global long *d_Superaccs
) {
    // Load a slice of X in WORK, using all available threads
	if ((offsetx == 0) && (incx == 1)) {  
		for (int k = 0; k < m; k += get_local_size(ROW_DIM)) {
			int col = k + get_local_id(ROW_DIM);
			if (col < m)
				work[col] = x[col];
		}
	} else {
		for (int k = 0; k < m; k += get_local_size(ROW_DIM)) {
			int col = k + get_local_id(ROW_DIM);
			if (col < m)
				work[col] = x[offsetx + incx * col];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); // sync group

    __global long *l_working = d_Superaccs + (get_global_id(ROW_DIM) + n * get_global_id(COL_DIM))* BIN_COUNT;
    for (uint i = 0; i < BIN_COUNT; i++) {
        l_working[i] = 0.0;
    }

    // Compute partial dot product
    if (get_global_id(ROW_DIM) < n) {
		if (offseta == 0) {  
			for (int k = 0; k < m; k++) {
                double xs, r;
			    xs = TwoProductFMA(a[lda * get_global_id(ROW_DIM) + k], alpha * work[k], &r);

			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
			}
		} else {
			for (int k = 0; k < m; k++) {
                double xs, r;
			    xs = TwoProductFMA(a[offseta + lda * get_global_id(ROW_DIM) + k], alpha * work[k], &r);

			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
			}
		}
	}

    // Store in Y (P columns per row)
	if (get_global_id(ROW_DIM) < n) {
		if ((offsety == 0) && (incy == 1)) {  
			if (beta == 0.0) {
				y[get_global_id(ROW_DIM)] = Round(l_working);
			} else if (beta == 1.0) {
                Accumulate(l_working, y[get_global_id(ROW_DIM)]);
				y[get_global_id(ROW_DIM)] = Round(l_working);
			} else {
                double xs, r;
                xs = TwoProductFMA(beta, y[get_global_id(ROW_DIM)], &r);
			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
				y[get_global_id(ROW_DIM)] = Round(l_working);
			}
		} else {
			if (beta == 0.0) {
				y[offsety + incy * get_global_id(ROW_DIM)] = Round(l_working);
			} else if (beta == 1.0) {
                Accumulate(l_working, y[offsety + incy * get_global_id(ROW_DIM)]);
				y[offsety + incy * get_global_id(ROW_DIM)] = Round(l_working);
			} else {
                double xs, r;
                xs = TwoProductFMA(beta, y[offsety + incy * get_global_id(ROW_DIM)], &r);
			    Accumulate(l_working, xs);
			    if (r != 0.0) {
				    Accumulate(l_working, r);
                }
				y[offsety + incy * get_global_id(ROW_DIM)] = Round(l_working);
			}
		}
	}
}


// Reduce M = get_global_size(0) rows of P values in matrix Y.
// Stores the result in first column of Y.
__kernel void gemv_reduce(
    uint m,
    uint p,
    __global long *d_Superaccs,
    __global double *y
) {
    int row = get_global_id(ROW_DIM);
    long suma[BIN_COUNT];
    for (int j = 0; j < BIN_COUNT; j++)
        suma[j] = 0;

    for(int col = 0; col < p; col++) {
        for (int j = 0; j < BIN_COUNT; j++) {
            suma[j] += d_Superaccs[(row + m * col) * BIN_COUNT + j];
        }
    }
    for (int j = 0; j < BIN_COUNT; j++)
        d_Superaccs[row * BIN_COUNT + j] = suma[j];
    y[row] = Round(d_Superaccs + row * BIN_COUNT);

    /*//Version with k x bin_count threads
    int lid = get_local_id(COL_DIM);

    if (lid < BIN_COUNT) {
        //y[get_global_id(ROW_DIM) + m * get_global_id(COL_DIM)] = Round(d_Superaccs + (row + m * col) * BIN_COUNT);
        long sum = 0;
        for (int j = 0; j < p; j++) {
            sum += d_Superaccs[(row + m * j) * BIN_COUNT + lid];
        }
        d_Superaccs[row * BIN_COUNT + lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // sync group

    if (lid == 0)
        y[row] = Round(d_Superaccs + row * BIN_COUNT);*/
}

