#ifndef OCLBLAS_H_
#define OCLBLAS_H_

#include "oclblas_types.h"
#include "oclblas_s.h"
#include "oclblas_d.h"
#include "oclblas_c.h"
#include "oclblas_z.h"

#ifdef __cplusplus
extern "C" {
#endif

oclblas_err_t oclblas_setup();
oclblas_err_t oclblas_setupEx(const unsigned int target_platform_no, const unsigned int target_device_type);
oclblas_err_t oclblas_setup_with_devices(const cl_platform_id *platform, const cl_uint num_devices, cl_device_id *devices);
oclblas_err_t oclblas_setup_with_context(const cl_platform_id *platform, const cl_uint num_devices, cl_device_id *devices, const cl_context *context);
oclblas_err_t oclblas_teardown();
cl_platform_id oclblas_get_platform();
cl_device_id *oclblas_get_devices();
cl_context oclblas_get_context();
cl_command_queue *oclblas_get_command_queues();
const char *oclblas_get_error_message(const int error_value);

void oclblas_xerbla(const char *srname, void *vinfo);

#ifdef __cplusplus
}
#endif

#endif // #ifndef OCLBLAS_H_
