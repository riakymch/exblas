#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(BENCHMARK_ATLAS) || defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_ACML) || defined(BENCHMARK_CUBLAS) || defined(BENCHMARK_MAGMA) || defined(BENCHMARK_OPENBLAS) || defined(BENCHMARK_OCLBLAS)
#include <cblas.h>
#include <clapack.h>
#endif

#if 0

#elif defined(BENCHMARK_MKL)
#include <mkl.h>
#define DCOPY() cblas_dcopy(N, X, incx, Y, incy)

#elif defined(BENCHMARK_AMDCLBLAS)
#include <clAmdBlas.h>
#define DCOPY() clAmdBlasDcopy(N, bufX, 0, incx, bufY, 0, incy, 1, &queue, 0, NULL, &event)

#elif defined(BENCHMARK_OCLBLAS)
#include "oclblas.h"
#define DCOPY() oclblas_dcopy(N, bufX, 0, incx, bufY, 0, incy, 1, &queue, 0, NULL, &event)

#elif defined(BENCHMARK_CUBLAS)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define DCOPY() cublasDcopy(handle, N, bufX, incx, bufY, incy)

#elif defined(BENCHMARK_MAGMA)
#include <magma.h>
#define DCOPY() printf("DCOPY IS NOT SUPPORTED\n")

#endif

#include "flops.h"
#include "benchmark.h"

int main(int argc, char *argv[])
{
  int ione = 1;
  double neg_one = D_NEG_ONE;
  double comp_time=0., perf, error;
  double work[1];
  int N, sizeX, sizeY;
#if defined(BENCHMARK_CUBLAS)
  cudaEvent_t start_tm, end_tm;
  float comp_time_f=0.f;
#else
  double start_tm, end_tm;
#endif

  const int incx        = (argc <= 1) ? 1   : atoi(argv[1]);
  const int incy        = (argc <= 2) ? 1   : atoi(argv[2]);
  const int max_size    = (argc <= 3) ? 512 : atoi(argv[3]);
  const int increment   = (argc <= 4) ? 1   : atoi(argv[4]);
  const int error_check = (argc <= 5) ? 0   : atoi(argv[5]);
  double *X, *Y, *Y2;
#ifdef BENCHMARK_MAGMA
#else
  X  = (double *)memalign(256, max_size*abs(incx)*sizeof(double));
  Y  = (double *)memalign(256, max_size*abs(incy)*sizeof(double));
  Y2 = (double *)memalign(256, max_size*max(abs(incx),abs(incy))*sizeof(double));
#endif

#if defined(BENCHMARK_AMDCLBLAS)
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx;
  cl_command_queue queue;
  cl_mem bufX, bufY;
  cl_event event = NULL;
  /* Setup OpenCL environment. */
  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);
  /* Setup clAmdBlas. */
  err = clAmdBlasSetup();
  if (err != CL_SUCCESS) {
    printf("clAmdBlasSetup() failed with %d\n", err);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 1;
  }
  // Create buffers
  bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*abs(incx)*sizeof(*X), NULL, &err);
  bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*abs(incy)*sizeof(*Y), NULL, &err);
  // Dummy
  N = min(512, max_size);
  DCOPY();
  err = clFinish(queue);
#elif defined(BENCHMARK_OCLBLAS)
  cl_int err;
  err = oclblas_setup();
  if (err != OCLBLAS_SUCCESS) {
    fprintf(stdout, "ERROR [%d in %s]: Failed to setup oclblas\n", __LINE__, __FILE__);
    return err;
  }
  cl_context ctx = oclblas_get_context();
  cl_command_queue queue = oclblas_get_command_queues()[0];
  cl_mem bufX, bufY;
  cl_event event = NULL;
  // Create buffers
  bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*abs(incx)*sizeof(*X), NULL, &err);
  bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*abs(incy)*sizeof(*Y), NULL, &err);
  // Dummy
  N = min(64, max_size);
  DCOPY();
  err = clFinish(queue);
#elif defined(BENCHMARK_CUBLAS)
  //cublasStatus stat;
  double *bufX, *bufY;
  // Initialize CUBLAS
  cublasHandle_t handle;
  cudaStream_t stream;
  cublasCreate(&handle);
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  cudaEventCreate(&start_tm);
  cudaEventCreate(&end_tm);
#endif

  int reset = TRUE;
  fprintf(stderr, "               N    \n");
  for (N = increment; N <= max_size; N += increment) {
    oclblas_dmake("GE", ' ', ' ', N*abs(incx), 1, Y2, 1, X, 1, &reset, 0.);
    oclblas_dmake("GE", ' ', ' ', N*abs(incy), 1, Y2, 1, Y, 1, &reset, 0.);
    sizeX = N * abs(incx);
    sizeY = N * abs(incy);
    memcpy(Y2, Y, sizeY*sizeof(*Y));

#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, sizeX*sizeof(*X), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0, sizeY*sizeof(*Y), Y, 0, NULL, NULL);
#elif defined(BENCHMARK_CUBLAS)
    // Prepare memory objects
    cudaMalloc((void**)(&bufX), sizeX*sizeof(*X));
    cudaMalloc((void**)(&bufY), sizeY*sizeof(*Y));
    cudaMemcpy(bufX, X, sizeX*sizeof(*X), cudaMemcpyHostToDevice);
    cudaMemcpy(bufY, Y, sizeY*sizeof(*Y), cudaMemcpyHostToDevice);
#endif

    printf("dcopy %2d %2d : %5d",
        incx, incy, N);

    double bytes = N * sizeof(double);
#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(start_tm, stream);
#else
    start_tm = oclblas_get_current_time();
#endif
    DCOPY();
#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
    err = clFinish(queue);
#elif defined(BENCHMARK_CUBLAS)
    cudaEventRecord(end_tm, stream);
    cudaEventSynchronize(end_tm);
    cudaEventElapsedTime(&comp_time_f, start_tm, end_tm);
    comp_time = comp_time_f * 1.e-3;
#else
    end_tm = oclblas_get_current_time();
    comp_time = end_tm - start_tm;
#endif
    perf = bytes / comp_time * 1.e-9;

    if (error_check) {
#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
      err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, sizeY*sizeof(*Y), Y, 0, NULL, NULL);
#elif defined(BENCHMARK_CUBLAS)
      cudaMemcpy(Y, bufY, sizeY*sizeof(*Y), cudaMemcpyDeviceToHost);
#elif defined(BENCHMARK_MAGMA)
#endif
      cblas_dcopy(N, X, incx, Y2, incy);
      blasf77_daxpy(&sizeY, &neg_one, Y, &ione, Y2, &ione);
#if defined(BENCHMARK_ACML)
#else
      error = fabs(lapackf77_dlange("M", &N, &ione, Y2, &ione, work));
#endif
      printf(" : %10.6lf sec %12.6lf GByte/s   %e [%s]", comp_time, perf, error, (error<D_THRESHOLD) ? "PASSED" : "FAILED");
    } else {
      printf(" : %10.6lf sec %12.6lf GByte/s   -", comp_time, perf);
    }

#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(start_tm, stream);
#else
    start_tm = oclblas_get_current_time();
#endif
    DCOPY();
#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
    err = clFinish(queue);
#endif
#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(end_tm, stream);
    cudaEventSynchronize(end_tm);
    cudaEventElapsedTime(&comp_time_f, start_tm, end_tm);
    comp_time = comp_time_f * 1.e-3;
#else
    end_tm = oclblas_get_current_time();
    comp_time = end_tm - start_tm;
#endif
    perf = bytes / comp_time * 1.e-9;
    printf(" : %10.6lf sec %12.6lf GByte/s\n", comp_time, perf);
    fflush(stdout);

#if defined(BENCHMARK_CUBLAS)
    cudaFree(bufX);
    cudaFree(bufY);
#endif
  }
  free(X); free(Y); free(Y2);
#if defined(BENCHMARK_AMDCLBLAS)
  clReleaseMemObject(bufX);
  clReleaseMemObject(bufY);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  clAmdBlasTeardown();
#elif defined(BENCHMARK_OCLBLAS)
  oclblas_teardown();
#elif defined(BENCHMARK_CUBLAS)
  cudaEventDestroy(end_tm);
  cudaEventDestroy(start_tm);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
#elif defined(BENCHMARK_MAGMA)
  cublasShutdown();
#endif

  return 0;
}
