#include <cstdio>
#include <string>
#include <CL/cl.h>
#include "oclblas_runtime.hpp"

oclblas_runtime *rt;

oclblas_err_t
oclblas_setup()
{
  rt = oclblas_runtime::instance();
  return rt->initialize();
}

oclblas_err_t
oclblas_setupEx(const unsigned int target_platform_no, const unsigned int target_device_type)
{
  rt = oclblas_runtime::instance();
  return rt->initialize(target_platform_no, target_device_type);
}

oclblas_err_t
oclblas_setup_with_devices(const cl_platform_id &platform, const cl_uint num_devices, cl_device_id *devices)
{
  rt = oclblas_runtime::instance();
  return rt->initialize(platform, num_devices, devices);
}

oclblas_err_t
oclblas_setup_with_context(const cl_platform_id &platform, const cl_uint num_devices, cl_device_id *devices, const cl_context *context)
{
  rt = oclblas_runtime::instance();
  return rt->initialize(platform, num_devices, devices, *context);
}

oclblas_err_t
oclblas_teardown()
{
  return rt->finalize();
}

cl_platform_id
oclblas_get_platform()
{
  return rt->getPlatform();
}

cl_uint
oclblas_get_num_devices()
{
  return rt->getNumDevices();
}

cl_device_id *
oclblas_get_devices()
{
  return rt->getDevices();
}

cl_context
oclblas_get_context()
{
  return rt->getContext();
}

cl_command_queue *
oclblas_get_command_queues()
{
  return rt->getCommandQueues();
}

const char *
oclblas_get_error_message(const int error_value)
{
  return rt->getErrorMessage(error_value).c_str();
}
