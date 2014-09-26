/* oclblas_sgemm.cpp
 *
 *  Purpose
 *  =======

 *  SGEMM  performs one of the matrix-matrix operations

 *     C := alpha*op( A )*op( B ) + beta*C,

 *  where  op( X ) is one of

 *     op( X ) = X   or   op( X ) = X**T,

 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

 *  Arguments
 *  ==========

 *  TRANSA - CHARACTER*1.
 *           On entry, TRANSA specifies the form of op( A ) to be used in
 *           the matrix multiplication as follows:

 *              TRANSA = 'N' or 'n',  op( A ) = A.

 *              TRANSA = 'T' or 't',  op( A ) = A**T.

 *              TRANSA = 'C' or 'c',  op( A ) = A**T.

 *           Unchanged on exit.

 *  TRANSB - CHARACTER*1.
 *           On entry, TRANSB specifies the form of op( B ) to be used in
 *           the matrix multiplication as follows:

 *              TRANSB = 'N' or 'n',  op( B ) = B.

 *              TRANSB = 'T' or 't',  op( B ) = B**T.

 *              TRANSB = 'C' or 'c',  op( B ) = B**T.

 *           Unchanged on exit.

 *  M      - INTEGER.
 *           On entry,  M  specifies  the number  of rows  of the  matrix
 *           op( A )  and of the  matrix  C.  M  must  be at least  zero.
 *           Unchanged on exit.

 *  N      - INTEGER.
 *           On entry,  N  specifies the number  of columns of the matrix
 *           op( B ) and the number of columns of the matrix C. N must be
 *           at least zero.
 *           Unchanged on exit.

 *  K      - INTEGER.
 *           On entry,  K  specifies  the number of columns of the matrix
 *           op( A ) and the number of rows of the matrix op( B ). K must
 *           be at least  zero.
 *           Unchanged on exit.

 *  ALPHA  - REAL            .
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.

 *  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
 *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
 *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
 *           part of the array  A  must contain the matrix  A,  otherwise
 *           the leading  k by m  part of the array  A  must contain  the
 *           matrix A.
 *           Unchanged on exit.

 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
 *           LDA must be at least  max( 1, m ), otherwise  LDA must be at
 *           least  max( 1, k ).
 *           Unchanged on exit.
 
 *  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
 *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
 *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
 *           part of the array  B  must contain the matrix  B,  otherwise
 *           the leading  n by k  part of the array  B  must contain  the
 *           matrix B.
 *           Unchanged on exit.
 
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
 *           LDB must be at least  max( 1, k ), otherwise  LDB must be at
 *           least  max( 1, n ).
 *           Unchanged on exit.
 
 *  BETA   - REAL            .
 *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
 *           supplied as zero then C need not be set on input.
 *           Unchanged on exit.
 
 *  C      - REAL             array of DIMENSION ( LDC, n ).
 *           Before entry, the leading  m by n  part of the array  C must
 *           contain the matrix  C,  except when  beta  is zero, in which
 *           case C need not be set on entry.
 *           On exit, the array  C  is overwritten by the  m by n  matrix
 *           ( alpha*op( A )*op( B ) + beta*C ).
 
 *  LDC    - INTEGER.
 *           On entry, LDC specifies the first dimension of C as declared
 *           in  the  calling  (sub)  program.   LDC  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 
 *  Further Details
 *  ===============
 
 *  Level 3 Blas routine.
 
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 
 *  =====================================================================  */
#include <cstdio>
#include <string>
#include "oclblas.h"
#include "oclblas_internal_interface.h"
#include "oclblas_tune_parameters.h"

oclblas_err_t
oclblas_sgemm_internal(
    const int nota, const int notb,
    const int m, const int n, const int k,
    const float alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const float beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    )
{
  if (m <= 0 || n <= 0 || k <= 0)
    return OCLBLAS_INVALID_FUNC_ARGUMENT;

  cl_int err;
  cl_event internal_events[2];
  size_t global_work_size[2];
  size_t local_work_size[2];
  size_t global_work_size_copya[2];
  size_t local_work_size_copya[2];
  size_t global_work_size_copyb[2];
  size_t local_work_size_copyb[2];
  static const cl_context context = oclblas_get_context();

  if (alpha == 0.) { // alpha equals zero
    static const cl_kernel sgemm_alpha_zero_kernel = oclblas_get_kernel("sgemm_alpha_zero");
    err  = clSetKernelArg(sgemm_alpha_zero_kernel, 0, sizeof(float) , &beta);
    err |= clSetKernelArg(sgemm_alpha_zero_kernel, 1, sizeof(cl_mem), &c);
    err |= clSetKernelArg(sgemm_alpha_zero_kernel, 2, sizeof(int)   , &offsetc);
    err |= clSetKernelArg(sgemm_alpha_zero_kernel, 3, sizeof(int)   , &ldc);
    //if (err != CL_SUCCESS) {
    //  fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to set kernel arguments\n", oclblas_get_error_message(err), __LINE__, __FILE__);
    //  return err;
    //}
    global_work_size[0] = m;
    global_work_size[1] = n;
    local_work_size[0] = 1;
    local_work_size[1] = 1;
    clEnqueueNDRangeKernel(command_queues[0], sgemm_alpha_zero_kernel, 2, NULL, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, events);
  } else { // sgemmK
    static const int m_wg = OCLBLAS_SGEMMK_M_WG;
    static const int n_wg = OCLBLAS_SGEMMK_N_WG;
    static const int k_wg = OCLBLAS_SGEMMK_K_WG;
    static const int m_wi = OCLBLAS_SGEMMK_M_WI;
    static const int n_wi = OCLBLAS_SGEMMK_N_WI;
    const int pm = (m+m_wg-1) / m_wg * m_wg;
    const int pn = (n+n_wg-1) / n_wg * n_wg;
    const int pk = (k+k_wg-1) / k_wg * k_wg;
#if defined(OCLBLAS_SGEMMK_USE_COL_LAYOUT_A) || defined(OCLBLAS_SGEMMK_USE_CBC_LAYOUT_A)
    const int pmk = pm;
#elif defined(OCLBLAS_SGEMMK_USE_RBC_LAYOUT_A)
    const int pmk = pk;
#endif
#if defined(OCLBLAS_SGEMMK_USE_COL_LAYOUT_B) || defined(OCLBLAS_SGEMMK_USE_CBC_LAYOUT_B)
    const int pnk = pn;
#elif defined(OCLBLAS_SGEMMK_USE_RBC_LAYOUT_B)
    const int pnk = pk;
#endif

    cl_mem bufa = oclblas_get_bufa(pm*pk*sizeof(float));
    cl_mem bufb = oclblas_get_bufb(pn*pk*sizeof(float));
    if (bufa == NULL || bufb == NULL) {
      fprintf(stderr, "ERROR [%d in %s]: Failed to create temporary buffers\n", __LINE__, __FILE__);
      return OCLBLAS_CREATE_BUFFER_FAILURE;
    }

    static cl_kernel sgemmK_kernel             = oclblas_get_kernel("sgemmK");
    static cl_kernel sgemmK_copyA_kernel       = oclblas_get_kernel("sgemmK_copyA");
    static cl_kernel sgemmK_copyA_trans_kernel = oclblas_get_kernel("sgemmK_copyA_trans");
    static cl_kernel sgemmK_copyB_kernel       = oclblas_get_kernel("sgemmK_copyB");
    static cl_kernel sgemmK_copyB_trans_kernel = oclblas_get_kernel("sgemmK_copyB_trans");
    cl_kernel *copyA_kernel;
    cl_kernel *copyB_kernel;
    if (nota) {
      copyA_kernel = &sgemmK_copyA_kernel;
      global_work_size_copya[0] = pm / OCLBLAS_SGEMMK_COPYA_M_WI;
      global_work_size_copya[1] = pk / OCLBLAS_SGEMMK_COPYA_K_WI;
      local_work_size_copya[0]  = OCLBLAS_SGEMMK_COPYA_M_WG / OCLBLAS_SGEMMK_COPYA_M_WI;
      local_work_size_copya[1]  = OCLBLAS_SGEMMK_COPYA_K_WG / OCLBLAS_SGEMMK_COPYA_K_WI;
    } else {
      copyA_kernel = &sgemmK_copyA_trans_kernel;
      global_work_size_copya[1] = pm / OCLBLAS_SGEMMK_COPYA_TRANS_M_WI;
      global_work_size_copya[0] = pk / OCLBLAS_SGEMMK_COPYA_TRANS_K_WI;
      local_work_size_copya[1]  = OCLBLAS_SGEMMK_COPYA_TRANS_M_WG / OCLBLAS_SGEMMK_COPYA_TRANS_M_WI;
      local_work_size_copya[0]  = OCLBLAS_SGEMMK_COPYA_TRANS_K_WG / OCLBLAS_SGEMMK_COPYA_TRANS_K_WI;
    }
    if (notb) {
      copyB_kernel = &sgemmK_copyB_trans_kernel;
      global_work_size_copyb[1] = pn / OCLBLAS_SGEMMK_COPYB_TRANS_N_WI;
      global_work_size_copyb[0] = pk / OCLBLAS_SGEMMK_COPYB_TRANS_K_WI;
      local_work_size_copyb[1]  = OCLBLAS_SGEMMK_COPYB_TRANS_N_WG / OCLBLAS_SGEMMK_COPYB_TRANS_N_WI;
      local_work_size_copyb[0]  = OCLBLAS_SGEMMK_COPYB_TRANS_K_WG / OCLBLAS_SGEMMK_COPYB_TRANS_K_WI;
    } else {
      copyB_kernel = &sgemmK_copyB_kernel;
      global_work_size_copyb[0] = pn / OCLBLAS_SGEMMK_COPYB_N_WI;
      global_work_size_copyb[1] = pk / OCLBLAS_SGEMMK_COPYB_K_WI;
      local_work_size_copyb[0]  = OCLBLAS_SGEMMK_COPYB_N_WG / OCLBLAS_SGEMMK_COPYB_N_WI;
      local_work_size_copyb[1]  = OCLBLAS_SGEMMK_COPYB_K_WG / OCLBLAS_SGEMMK_COPYB_K_WI;
    }

    err  = clSetKernelArg(*copyA_kernel, 0, sizeof(int)   , &m);
    err |= clSetKernelArg(*copyA_kernel, 1, sizeof(int)   , &k);
    err |= clSetKernelArg(*copyA_kernel, 2, sizeof(int)   , &lda);
    err |= clSetKernelArg(*copyA_kernel, 3, sizeof(int)   , &pmk);
    err |= clSetKernelArg(*copyA_kernel, 4, sizeof(int)   , &offseta);
    err |= clSetKernelArg(*copyA_kernel, 5, sizeof(cl_mem), &a);
    err |= clSetKernelArg(*copyA_kernel, 6, sizeof(cl_mem), &bufa);

    err |= clSetKernelArg(*copyB_kernel, 0, sizeof(int)   , &n);
    err |= clSetKernelArg(*copyB_kernel, 1, sizeof(int)   , &k);
    err |= clSetKernelArg(*copyB_kernel, 2, sizeof(int)   , &ldb);
    err |= clSetKernelArg(*copyB_kernel, 3, sizeof(int)   , &pnk);
    err |= clSetKernelArg(*copyB_kernel, 4, sizeof(int)   , &offsetb);
    err |= clSetKernelArg(*copyB_kernel, 5, sizeof(cl_mem), &b);
    err |= clSetKernelArg(*copyB_kernel, 6, sizeof(cl_mem), &bufb);

    err |= clSetKernelArg(sgemmK_kernel,  0, sizeof(int)   , &m);
    err |= clSetKernelArg(sgemmK_kernel,  1, sizeof(int)   , &n);
    err |= clSetKernelArg(sgemmK_kernel,  2, sizeof(int)   , &pk);
    err |= clSetKernelArg(sgemmK_kernel,  3, sizeof(cl_mem), &bufa);
    err |= clSetKernelArg(sgemmK_kernel,  4, sizeof(cl_mem), &bufb);
    err |= clSetKernelArg(sgemmK_kernel,  5, sizeof(cl_mem), &c);
    err |= clSetKernelArg(sgemmK_kernel,  6, sizeof(int)   , &pm);
    err |= clSetKernelArg(sgemmK_kernel,  7, sizeof(int)   , &pn);
    err |= clSetKernelArg(sgemmK_kernel,  8, sizeof(int)   , &ldc);
    err |= clSetKernelArg(sgemmK_kernel,  9, sizeof(int)   , &offsetc);
    err |= clSetKernelArg(sgemmK_kernel, 10, sizeof(float) , &alpha);
    err |= clSetKernelArg(sgemmK_kernel, 11, sizeof(float) , &beta);

    //if (err != CL_SUCCESS) {
    //  fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to set kernel arguments\n", oclblas_get_error_message(err), __LINE__, __FILE__);
    //  return err;
    //}

    clEnqueueNDRangeKernel(command_queues[0], *copyA_kernel, 2, NULL, global_work_size_copya, local_work_size_copya, num_events_in_wait_list, event_wait_list, &internal_events[0]);
    clEnqueueNDRangeKernel(command_queues[0], *copyB_kernel, 2, NULL, global_work_size_copyb, local_work_size_copyb, 1, &internal_events[0], &internal_events[1]);
    global_work_size[0] = pm/m_wi;
    global_work_size[1] = pn/n_wi;
    local_work_size[0] = m_wg/m_wi;
    local_work_size[1] = n_wg/n_wi;
    clEnqueueNDRangeKernel(command_queues[0], sgemmK_kernel, 2, NULL, global_work_size, local_work_size, 1, &internal_events[1], events);
  }
}

extern "C"
oclblas_err_t
oclblas_sgemm(
    oclblas_order_t order,
    oclblas_trans_t transa, oclblas_trans_t transb,
    int m, int n, int k,
    float alpha,
    const cl_mem a, int offseta, int lda,
    const cl_mem b, int offsetb, int ldb,
    float beta,
    cl_mem c, int offsetc, int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    )
{
  if (order == OclblasRowMajor) {
    fprintf(stderr, "ERROR [%d in %s]: Row-major matrix storage order is currently not supported\n", __LINE__, __FILE__);
    return OCLBLAS_INVALID_FUNCTION_CALL;
  }

  /* Local variables */
  static int info;
  static int nota, notb;
  static int ncola;
  static int nrowa, nrowb;

  /*
   *     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
   *     transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
   *     and  columns of  A  and the  number of  rows  of  B  respectively.
   */

  /* Function Body */
  nota = (transa == OclblasNoTrans);
  notb = (transb == OclblasNoTrans);
  if (nota) {
    nrowa = m;
    ncola = k;
  } else {
    nrowa = k;
    ncola = m;
  }
  if (notb) {
    nrowb = k;
  } else {
    nrowb = n;
  }

  /* Test the input parameters. */
  info = 0;
  if (!nota && transa != OclblasConjTrans && transa != OclblasTrans) {
    info = 2;
  } else if (!notb && transb != OclblasConjTrans && transb != OclblasTrans) {
    info = 3;
  } else if (m < 0) {
    info = 4;
  } else if (n < 0) {
    info = 5;
  } else if (k < 0) {
    info = 6;
  } else if (lda < max(1,nrowa)) {
    info = 10;
  } else if (ldb < max(1,nrowb)) {
    info = 13;
  } else if (ldc < max(1,m)) {
    info = 16;
  }
  if (info != 0) {
    oclblas_xerbla(std::string("oclblas_sgemm").c_str(), &info);
    return OCLBLAS_INVALID_FUNC_ARGUMENT;
  }

  /* Quick return if possible. */
  if (m == 0 || n == 0 || (alpha == 0.f || k == 0) && beta == 1.f) {
    return OCLBLAS_SUCCESS;
  }

  static const int bsz = GEMM_BSZ;
  switch (GEMM_ALGO) {
    case GEMM_ALGO_MM:
      oclblas_sgemm_internal(nota, notb, m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc, num_command_queues, command_queues, num_events_in_wait_list, event_wait_list, events);
      break;
    case GEMM_ALGO_MP:
      break;
    case GEMM_ALGO_PM:
      break;
    case GEMM_ALGO_PP:
      break;
  }

  return OCLBLAS_SUCCESS;
} // End of oclblas_sgemm

extern "C"
oclblas_err_t
oclblas_sgemm0(
    const oclblas_order_t order,
    const oclblas_trans_t transa, const oclblas_trans_t transb,
    const int m, const int n, int k,
    const float alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    const float beta,
    cl_mem c, int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    )
{
  return oclblas_sgemm(order, transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc, num_command_queues, command_queues, num_events_in_wait_list, event_wait_list, events);
}
