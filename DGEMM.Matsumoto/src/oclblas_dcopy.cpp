/* oclblas_dcopy.cpp
 *
 *  Purpose
 *  =======
 
 *     DCOPY copies a vector, x, to a vector, y.
 *     uses unrolled loops for increments equal to one.
 
 *  Further Details
 *  ===============
 
 *     jack dongarra, linpack, 3/11/78.
 *     modified 12/3/93, array(1) declarations changed to array(*)
 
 *  =====================================================================  */
#include <cstdio>
#include <string>
#include "oclblas.h"
#include "oclblas_internal_interface.h"
#include "oclblas_tune_parameters.h"

extern "C"
oclblas_err_t
oclblas_dcopy(
    int n,
    const cl_mem x, const int offsetx, const int incx,
    cl_mem y, const int offsety, const int incy,
    cl_uint num_command_queues, cl_command_queue *command_queues,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    )
{
  /* Local variables */
  static int info;

  /* Test the input parameters. */
  info = 0;
  if (incx == 0) {
    info = 4;
  } else if (incy == 0) {
    info = 7;
  }
  if (info != 0) {
    oclblas_xerbla(std::string("oclblas_dcopy").c_str(), &info);
    return OCLBLAS_INVALID_FUNC_ARGUMENT;
  }

  /* Function Body */
  if (n <= 0) {
    return 0;
  }

  cl_int err;
  size_t global_work_size[2];
  size_t local_work_size[2];
  static const cl_kernel dcopy_kernel = oclblas_get_kernel("dcopy");

  err  = clSetKernelArg(dcopy_kernel,  0, sizeof(int),    &n);
  err |= clSetKernelArg(dcopy_kernel,  1, sizeof(cl_mem), &x);
  err  = clSetKernelArg(dcopy_kernel,  2, sizeof(int),    &offsetx);
  err  = clSetKernelArg(dcopy_kernel,  3, sizeof(int),    &incx);
  err |= clSetKernelArg(dcopy_kernel,  4, sizeof(cl_mem), &y);
  err  = clSetKernelArg(dcopy_kernel,  5, sizeof(int),    &offsety);
  err  = clSetKernelArg(dcopy_kernel,  6, sizeof(int),    &incy);
  if (err != CL_SUCCESS) {
    fprintf(stdout, "ERROR [%d in %s]: Failed to set kernel arguments\n", __LINE__, __FILE__);
    return err;
  }
  global_work_size[0] = 1;
  global_work_size[1] = 1;
  local_work_size[0] = 1;
  local_work_size[1] = 1;
  clEnqueueNDRangeKernel(command_queues[0], dcopy_kernel, 2, NULL, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, events);
  return 0;
} /* End of oclblas_dcopy function */

extern "C"
oclblas_err_t
oclblas_dcopy0(
    int n,
    const cl_mem x, const int incx,
    cl_mem y, const int incy,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    )
{
  return oclblas_dcopy(n, x, 0, incx, y, 0, incy, num_command_queues, command_queues, num_events_in_wait_list, event_wait_list, events);
}
