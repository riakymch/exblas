#ifndef OCLBLAS_D_H_
#define OCLBLAS_D_H_

#include "oclblas_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Level-1 BLAS functions
oclblas_err_t
oclblas_dcopy(
    int n,
    const cl_mem x, const int offsetx, const int incx,
    cl_mem y, const int offsety, const int incy,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dcopy0(
    int n,
    const cl_mem x, const int incx,
    cl_mem y, const int incy,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

// Level-2 BLAS functions

// Level-3 BLAS functions
oclblas_err_t
oclblas_dgemm(
    const oclblas_order_t order,
    const oclblas_trans_t transa, const oclblas_trans_t transb,
    const int m, const int n, const int k,
    const double alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const double beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dgemm0(
    const oclblas_order_t order,
    const oclblas_trans_t transa, const oclblas_trans_t transb,
    const int m, const int n, const int k,
    const double alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    const double beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_dsymm(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const int m, const int n,
    const double alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const double beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dsymm0(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const int m, const int n,
    const double alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    double beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_dsyrk(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const double alpha,
    const cl_mem a, const int offseta, const int lda,
    const double beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dsyrk0(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const double alpha,
    const cl_mem a, const int lda,
    const double beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_dsyr2k(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const double alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const double beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dsyr2k0(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const double alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    const double beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_dtrmm(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const oclblas_trans_t transa, const oclblas_diag_t diag,
    const int m, const int n,
    double alpha,
    const cl_mem a, const int offseta, const int lda,
    cl_mem b, const int offsetb, const int ldb,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dtrmm0(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const oclblas_trans_t transa, const oclblas_diag_t diag,
    const int m, const int n,
    double alpha,
    const cl_mem a, const int lda,
    cl_mem b, const int ldb,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_dtrsm(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const oclblas_trans_t transa, const oclblas_diag_t diag,
    const int m, const int n,
    const double alpha,
    const cl_mem a, const int offseta, const int lda,
    cl_mem b, const int offsetb, const int ldb,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_dtrsm0(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const oclblas_trans_t transa, const oclblas_diag_t diag,
    const int m, const int n,
    const double alpha,
    const cl_mem a, const int lda,
    cl_mem b, const int ldb,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

#ifdef __cplusplus
}
#endif

#endif // #ifndef OCLBLAS_D_H_
