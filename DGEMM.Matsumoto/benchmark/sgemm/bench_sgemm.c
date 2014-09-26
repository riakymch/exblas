#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(BENCHMARK_ATLAS) || defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_ACML) || defined(BENCHMARK_CUBLAS) || defined(BENCHMARK_MAGMA) || defined(BENCHMARK_OPENBLAS) || defined(BENCHMARK_OCLBLAS)
#include <cblas.h>
#include <clapack.h>
#endif

#if defined(BENCHMARK_ATLAS) || defined(BENCHMARK_OPENBLAS)
#define SGEMM_COL() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define SGEMM_ROW() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

#elif defined(BENCHMARK_MKL)
#include <mkl.h>
#define SGEMM_COL() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define SGEMM_ROW() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

#elif defined(BENCHMARK_ACML)
#include <acml.h>
#define SGEMM_COL() sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define SGEMM_ROW() sgemm(transb, transa, N, K, K, alpha, B, ldb, A, lda, beta, C, ldc)

#elif defined(BENCHMARK_AMDCLBLAS)
#include <clAmdBlas.h>
#define SGEMM_COL() clAmdBlasSgemm(order, transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc, 1, &queue, 0, NULL, &event)
#define SGEMM_ROW() clAmdBlasSgemm(order, transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc, 1, &queue, 0, NULL, &event)

#elif defined(BENCHMARK_OCLBLAS)
#include "oclblas.h"
#define SGEMM_COL() oclblas_sgemm(order, transa, transb, M, N, K, alpha, bufA, 0, lda, bufB, 0, ldb, beta, bufC, 0, ldc, 1, &queue, 0, NULL, &event)
#define SGEMM_ROW() oclblas_sgemm(order, transb, transa, N, M, K, alpha, bufB, 0, ldb, bufA, 0, lda, beta, bufC, 0, ldc, 1, &queue, 0, NULL, &event)

#elif defined(BENCHMARK_CUBLAS)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define SGEMM_COL() cublasSgemm(handle, transa, transb, M, N, K, &alpha, bufA, lda, bufB, ldb, &beta, bufC, ldc);
#define SGEMM_ROW() cublasSgemm(handle, transb, transa, N, M, K, &alpha, bufB, ldb, bufA, lda, &beta, bufC, ldc);

#elif defined(BENCHMARK_MAGMA)
#include <magma.h>
#define SGEMM_COL() magmablas_sgemm(transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc);
#define SGEMM_ROW() magmablas_sgemm(transb, transa, N, M, K, alpha, bufB, ldb, bufA, lda, beta, bufC, ldc);

#endif

#include "flops.h"
#include "benchmark.h"

int main(int argc, char *argv[])
{
  static const float alpha =  0.27f;
  static const float beta  = -0.49f;
  int ione = 1;
  float neg_one = S_NEG_ONE;
  double comp_time, perf, error;
  float work[1];
  int M, N, K, lda, ldb, ldc, sizeA, sizeB, sizeC;
#if defined(BENCHMARK_CUBLAS)
  cudaEvent_t start_tm, end_tm;
  float s_comp_time;
#elif defined(BENCHMARK_MAGMA)
  magma_timestr_t start_tm, end_tm;
#else
  double start_tm, end_tm;
#endif

#if defined(BENCHMARK_ATLAS) || defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_ACML) || defined(BENCHMARK_CUBLAS) || defined(BENCHMARK_MAGMA) || defined(BENCHMARK_OPENBLAS) || defined(BENCHMARK_OCLBLAS)
  const enum CBLAS_ORDER     Order  = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const enum CBLAS_TRANSPOSE TransA = (argc <= 2) ? CblasNoTrans  : ((atoi(argv[2])==0) ? CblasNoTrans  : CblasTrans);
  const enum CBLAS_TRANSPOSE TransB = (argc <= 3) ? CblasNoTrans  : ((atoi(argv[3])==0) ? CblasNoTrans  : CblasTrans);
#elif defined(BENCHMARK_MKL)
  const CBLAS_ORDER     Order  = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const CBLAS_TRANSPOSE TransA = (argc <= 2) ? CblasNoTrans  : ((atoi(argv[2])==0) ? CblasNoTrans  : CblasTrans);
  const CBLAS_TRANSPOSE TransB = (argc <= 3) ? CblasNoTrans  : ((atoi(argv[3])==0) ? CblasNoTrans  : CblasTrans);
#endif

  const int max_size    = (argc <= 4) ? 512 : atoi(argv[4]);
  const int increment   = (argc <= 5) ? 1   : atoi(argv[5]);
  const int error_check = (argc <= 6) ? 0   : atoi(argv[6]);
  const int alloc_size  = max_size + 32;
  float *A, *B, *C, *C2;
#ifdef BENCHMARK_MAGMA
  magma_malloc_cpu((void**) &A , alloc_size*alloc_size*sizeof(float));
  magma_malloc_cpu((void**) &B , alloc_size*alloc_size*sizeof(float));
  magma_malloc_cpu((void**) &C , alloc_size*alloc_size*sizeof(float));
  magma_malloc_cpu((void**) &C2, alloc_size*alloc_size*sizeof(float));
#else
  A  = (float *)memalign(256, alloc_size*alloc_size*sizeof(float));
  B  = (float *)memalign(256, alloc_size*alloc_size*sizeof(float));
  C  = (float *)memalign(256, alloc_size*alloc_size*sizeof(float));
  C2 = (float *)memalign(256, alloc_size*alloc_size*sizeof(float));
#endif

#if defined(BENCHMARK_AMDCLBLAS)
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx;
  cl_command_queue queue;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  const clAmdBlasOrder     order  = (Order  == CblasColMajor) ? clAmdBlasColumnMajor : clAmdBlasRowMajor;
  const clAmdBlasTranspose transa = (TransA == CblasNoTrans ) ? clAmdBlasNoTrans     : clAmdBlasTrans;
  const clAmdBlasTranspose transb = (TransB == CblasNoTrans ) ? clAmdBlasNoTrans     : clAmdBlasTrans;
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
  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY , alloc_size*alloc_size*sizeof(*A), NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY , alloc_size*alloc_size*sizeof(*B), NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, alloc_size*alloc_size*sizeof(*C), NULL, &err);
  // Dummy
  M = N = K = lda = ldb = ldc = min(512, max_size);
  if (Order == CblasColMajor) {
    SGEMM_COL();
  } else {
    SGEMM_ROW();
  }
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
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  const oclblas_order_t order  = (Order  == CblasColMajor) ? OclblasColMajor : OclblasRowMajor;
  const oclblas_trans_t transa = (TransA == CblasNoTrans ) ? OclblasNoTrans : OclblasTrans;
  const oclblas_trans_t transb = (TransB == CblasNoTrans ) ? OclblasNoTrans : OclblasTrans;
  // Create buffers
  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  alloc_size*alloc_size*sizeof(*A), NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  alloc_size*alloc_size*sizeof(*B), NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, alloc_size*alloc_size*sizeof(*C), NULL, &err);
  // Dummy
  M = N = K = lda = ldb = ldc = min(64, max_size);
  if (Order == CblasColMajor) {
    SGEMM_COL();
  } else {
    SGEMM_ROW();
  }
#elif defined(BENCHMARK_CUBLAS)
  const cublasOperation_t transa = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t transb = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  //cublasStatus stat;
  float *bufA, *bufB, *bufC;
  // Initialize CUBLAS
  cublasHandle_t handle;
  cudaStream_t stream;
  cublasCreate(&handle);
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  cudaEventCreate(&start_tm);
  cudaEventCreate(&end_tm);
#elif defined(BENCHMARK_MAGMA)
  const char transa = (TransA == CblasNoTrans) ? MagmaNoTrans : MagmaTrans;
  const char transb = (TransB == CblasNoTrans) ? MagmaNoTrans : MagmaTrans;
  //cublasStatus stat;
  float *bufA, *bufB, *bufC;
  // Initialize CUBLAS
  if(CUBLAS_STATUS_SUCCESS != cublasInit()) {
    fprintf(stderr, "ERROR [%d in %s]: Failed to initialized cublas\n", __LINE__, __FILE__);
    exit(1);
  }

#elif defined(BENCHMARK_ACML)
  const char transa = (TransA == CblasNoTrans) ? 'N' : 'T';
  const char transb = (TransB == CblasNoTrans) ? 'N' : 'T';
#endif

  int reset = TRUE;
  fprintf(stderr, "               M    N    K\n");
  for (M = increment; M <= max_size; M += increment) {
  //for (M = max_size; M >= 1; M -= increment) {
    N = K = M;
    //K = 128;
    if (Order == CblasColMajor) {
      lda = (TransA == CblasTrans) ? K : M;
      ldb = (TransB == CblasTrans) ? N : K;
      ldc = M;
      oclblas_smake("GE", ' ', ' ', M, K, C2, lda, A, lda, &reset, 0.f);
      oclblas_smake("GE", ' ', ' ', K, N, C2, ldb, B, ldb, &reset, 0.f);
      oclblas_smake("GE", ' ', ' ', M, N, C2, ldc, C, ldc, &reset, 0.f);
    } else {
      lda = (TransA == CblasTrans) ? M : K;;
      ldb = (TransB == CblasTrans) ? K : N;
      ldc = N;
      oclblas_smake("GE", ' ', ' ', K, M, C2, lda, A, lda, &reset, 0.f);
      oclblas_smake("GE", ' ', ' ', N, K, C2, ldb, B, ldb, &reset, 0.f);
      oclblas_smake("GE", ' ', ' ', N, M, C2, ldc, C, ldc, &reset, 0.f);
    }
    sizeA = M*K;
    sizeB = K*N;
    sizeC = M*N;
    memcpy(C2, C, sizeC*sizeof(*C));

#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeA*sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeB*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, sizeC*sizeof(*C), C, 0, NULL, NULL);
#elif defined(BENCHMARK_CUBLAS)
    cudaMalloc((void**)(&bufA), sizeA*sizeof(*A));
    cudaMalloc((void**)(&bufB), sizeB*sizeof(*B));
    cudaMalloc((void**)(&bufC), sizeC*sizeof(*C));
    cudaMemcpy(bufA, A, sizeA*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(bufB, B, sizeB*sizeof(*B), cudaMemcpyHostToDevice);
    cudaMemcpy(bufC, C, sizeC*sizeof(*C), cudaMemcpyHostToDevice);
#elif defined(BENCHMARK_MAGMA)
    magma_malloc((void**)(&bufA), sizeA*sizeof(*A));
    magma_malloc((void**)(&bufB), sizeB*sizeof(*B));
    magma_malloc((void**)(&bufC), sizeC*sizeof(*C));
    magma_ssetmatrix(M, K, A, lda, bufA, lda);
    magma_ssetmatrix(K, N, B, ldb, bufB, ldb);
    magma_ssetmatrix(M, N, C, ldc, bufC, ldc);
#endif

    printf("sgemm %c%c%c : %4d %4d %4d",
        (Order == CblasColMajor) ? 'C' : 'R',
        (TransA == CblasNoTrans) ? 'N' : 'T',
        (TransB == CblasNoTrans) ? 'N' : 'T',
        M, N, K);

    double flops = FLOPS_SGEMM(M, N, K);
#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(start_tm, stream);
#elif defined(BENCHMARK_MAGMA)
    start_tm = get_current_time();
#else
    start_tm = oclblas_get_current_time();
#endif
    if (Order == CblasColMajor) {
      SGEMM_COL();
    } else {
      SGEMM_ROW();
    }
#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
    err = clFinish(queue);
#endif
#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(end_tm, stream);
    cudaEventSynchronize(end_tm);
    cudaEventElapsedTime(&s_comp_time, start_tm, end_tm);
    comp_time = s_comp_time * 1.e-3;
#elif defined(BENCHMARK_MAGMA)
    end_tm = get_current_time();
    comp_time = GetTimerValue(start_tm, end_tm) * 1.e-3;
#else
    end_tm = oclblas_get_current_time();
    comp_time = end_tm - start_tm;
#endif
    perf = flops / comp_time * 1.e-9;

    if (error_check) {
#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
      err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeC*sizeof(*C), C, 0, NULL, NULL);
#elif defined(BENCHMARK_CUBLAS)
      cudaMemcpy(C, bufC, sizeC*sizeof(*C), cudaMemcpyDeviceToHost);
#elif defined(BENCHMARK_MAGMA)
      magma_sgetmatrix(M, N, bufC, ldc, C, ldc);
#endif
      cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C2, ldc);
      blasf77_saxpy(&sizeC, &neg_one, C, &ione, C2, &ione);
#if defined(BENCHMARK_ACML)
      error = fabs(lapackf77_slange("M", &M, &N, C2, &ldc, work, ione));
#else
      error = fabs(lapackf77_slange("M", &M, &N, C2, &ldc, work));
#endif
      printf(" : %10.6lf sec %12.6lf GFlop/s   %e [%s]", comp_time, perf, error, (error<S_THRESHOLD) ? "PASSED" : "FAILED");
    } else {
      printf(" : %10.6lf sec %12.6lf GFlop/s   - ", comp_time, perf);
    }

#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(start_tm, stream);
#elif defined(BENCHMARK_MAGMA)
    start_tm = get_current_time();
#else
    start_tm = oclblas_get_current_time();
#endif
    if (Order == CblasColMajor) {
      SGEMM_COL();
    } else {
      SGEMM_ROW();
    }
#if defined(BENCHMARK_AMDCLBLAS) || defined(BENCHMARK_OCLBLAS)
    err = clFinish(queue);
#endif
#if defined(BENCHMARK_CUBLAS)
    cudaEventRecord(end_tm, stream);
    cudaEventSynchronize(end_tm);
    cudaEventElapsedTime(&s_comp_time, start_tm, end_tm);
    comp_time = s_comp_time * 1.e-3;
#elif defined(BENCHMARK_MAGMA)
    end_tm = get_current_time();
    comp_time = GetTimerValue(start_tm, end_tm) * 1.e-3;
#else
    end_tm = oclblas_get_current_time();
    comp_time = end_tm - start_tm;
#endif
    perf = flops / comp_time * 1.e-9;
    printf(": %10.6lf sec %12.6lf GFlop/s\n", comp_time, perf);
    fflush(stdout);

#if defined(BENCHMARK_CUBLAS)
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(bufC);
#elif defined(BENCHMARK_MAGMA)
    magma_free(bufA);
    magma_free(bufB);
    magma_free(bufC);
#endif
  }
  free(A); free(B); free(C); free(C2);
#if defined(BENCHMARK_AMDCLBLAS)
  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);
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
