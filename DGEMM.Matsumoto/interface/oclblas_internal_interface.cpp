#include <string>
#include <CL/cl.h>
#include "oclblas_runtime.hpp"
#include "oclblas_internal_interface.h"

extern oclblas_runtime *rt;

cl_kernel oclblas_get_kernel(const char *kernel_name)
{
  return rt->getKernel(std::string(kernel_name));
}

cl_mem oclblas_get_bufa(const size_t required_bufa_size)
{
  return rt->getBufA(required_bufa_size);
}

cl_mem oclblas_get_bufb(const size_t required_bufb_size)
{
  return rt->getBufB(required_bufb_size);
}
