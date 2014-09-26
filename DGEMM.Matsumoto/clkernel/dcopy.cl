#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel __attribute__((vec_type_hint(double)))
  void dcopy(
      const int n,
      __global double const * restrict x, const int offsetx, const int incx,
      __global double *y, const int offsety, const int incy
      )
{
  x += offsetx;
  y += offsety;

  if (incx == 1 && incy == 1) {
    /* code for both increments equal to 1 */

    /* clean-up loop */
    int m = n % 7;
    if (m != 0) {
      for (int i = 0; i < m; ++i) {
        y[i] = x[i];
      }
      if (n < 7) {
        return;
      }
    }
    int mp1 = m + 1;
    for (int i = m; i < n; i += 7) {
      y[i] = x[i];
      y[i + 1] = x[i + 1];
      y[i + 2] = x[i + 2];
      y[i + 3] = x[i + 3];
      y[i + 4] = x[i + 4];
      y[i + 5] = x[i + 5];
      y[i + 6] = x[i + 6];
    }
  } else {
    /* code for unequal increments or equal increments */
    /*   not equal to 1 */
    int ix = 0;
    int iy = 0;
    if (incx < 0) {
      ix = (-(n) + 1) * incx;
    }
    if (incy < 0) {
      iy = (-(n) + 1) * incy;
    }
    for (int i = 0; i < n; ++i) {
      y[iy] = x[ix];
      ix += incx;
      iy += incy;
    }
  }
}
