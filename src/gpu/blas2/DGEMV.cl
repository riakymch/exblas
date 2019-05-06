
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable  //For double precision numbers
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifdef NVIDIA
  #pragma OPENCL EXTENSION cl_nv_pragma_unroll         : enable
#endif

#define ROW_DIM 0
#define COL_DIM 1

// P threads per row compute 1/P-th of each dot product.
// WORK has N/P entries.
__kernel void dgemv(
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
    __local double *work
) {
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

    // Compute partial dot product
    double sum = 0.0;
    if (get_global_id(ROW_DIM) < m) {
		if (offseta == 0) {  
			for (int k = 0; k < ncols; k++) {
			    sum += alpha * a[get_global_id(ROW_DIM) + lda * (col0 + k)] * work[k];
			}
		} else {
			for (int k = 0; k < ncols; k++) {
			    sum += alpha * a[offseta + get_global_id(ROW_DIM) + lda * (col0 + k)] * work[k];
			}
		}
    }

    // Store in Y (P columns per row)
    if (get_global_id(ROW_DIM) < m) {
        if ((offsety == 0) && (incy == 1)) {
			if (beta == 0.0) {
				y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] = sum;
			} else if (beta == 1.0) {
				y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] += sum;
			} else {
				y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] += beta * y[get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM)] + sum;
			}
        } else {
			if (beta == 0.0) {
				y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] = sum;
			} else if (beta == 1.0) {
				y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] += sum;
			} else {
				y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] += beta * y[offsety + incy * (get_global_id(ROW_DIM) + lda * get_global_id(COL_DIM))] + sum;
			}
		}
	}
}


// P threads per row compute 1/P-th of each ddot product.
// WORK has N/P entries.
__kernel void dgemvT(
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
    __local double *work
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

    // Compute partial ddot product
    double sum = 0.0;
    if (get_global_id(ROW_DIM) < n) {
		if (offseta == 0) {  
			for (int k = 0; k < m; k++) {
				sum += alpha * a[lda * get_global_id(ROW_DIM) + k] * work[k];
			}
		} else {
			for (int k = 0; k < m; k++) {
				sum += alpha * a[offseta + lda * get_global_id(ROW_DIM) + k] * work[k];
				////sum += alpha * a[offseta + get_global_id(ROW_DIM) + k] * work[k];
			}
		}
	}

    // Store in Y (P columns per row)
	if (get_global_id(ROW_DIM) < n) {
		if ((offsety == 0) && (incy == 1)) {  
			if (beta == 0.0) {
				y[get_global_id(ROW_DIM)] = sum;
			} else if (beta == 1.0) {
				y[get_global_id(ROW_DIM)] += sum;
			} else {
				y[get_global_id(ROW_DIM)] = sum + beta * y[get_global_id(ROW_DIM)];
			}
		} else {
			if (beta == 0.0) {
				y[offsety + incy * get_global_id(ROW_DIM)] = sum;
			} else if (beta == 1.0) {
				y[offsety + incy * get_global_id(ROW_DIM)] += sum;
			} else {
				y[offsety + incy * get_global_id(ROW_DIM)] = sum + beta * y[offsety + incy * get_global_id(ROW_DIM)];
			}
		}
	}
}


// Reduce M = get_global_size(0) rows of P values in matrix Y.
// Stores the result in first column of Y.
__kernel void gemv_reduce(
    __global double *y,
    int m,
    int p
) {
    int row = get_global_id(0);

    double sum = 0.0;
    for (int col = 0; col < p; col++)
        sum += y[row + m * col];
    y[row] = sum;
}
