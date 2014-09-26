#ifndef OCLBLAS_RUNTIME_HPP_
#define OCLBLAS_RUNTIME_HPP_

#include <map>
#include <string>
#include <vector>
#include <CL/cl.h>
#include "oclblas.h"

class oclblas_runtime
{
  private:
    cl_platform_id    oclblas_platform;
    cl_uint           oclblas_num_devices;
    cl_device_id     *oclblas_devices;
    cl_context        oclblas_context;
    cl_command_queue *oclblas_command_queues;
    std::string       oclblas_kernel_dir;
    unsigned int platform_no;
    unsigned int device_type;
    size_t bufa_size, bufb_size;
    cl_mem bufa, bufb;

    bool initialized;
    bool creating_kernels;
    oclblas_err_t oclblas_err;
    
    std::map<std::string, cl_program> program_pool;
    std::map<std::string, cl_kernel> kernel_pool;
  public:
    enum sizes { MAX_INFO_PARAM_SIZE=1024 };
    oclblas_runtime();
    oclblas_runtime(const bool _creating_kernels);
    oclblas_runtime(const bool _creating_kernels, const std::string _oclblas_kernel_dir);
    static oclblas_runtime * instance() {
      static oclblas_runtime rt;
      return &rt;
    }
    static oclblas_runtime * instance(const bool _creating_kernels) {
      static oclblas_runtime rt(_creating_kernels);
      return &rt;
    }
    static oclblas_runtime * instance(const bool _creating_kernels, const std::string _oclblas_kernel_dir) {
      static oclblas_runtime rt(_creating_kernels, _oclblas_kernel_dir);
      return &rt;
    }
    oclblas_err_t initialize();
    oclblas_err_t initialize(const unsigned int target_platform_no, const unsigned int target_device_type);
    oclblas_err_t initialize(const cl_platform_id &platform, const cl_uint num_devices, cl_device_id *devices);
    oclblas_err_t initialize(const cl_platform_id &platform, const cl_uint num_devices, cl_device_id *devices, const cl_context &context);
    bool is_initialized() { return initialized; }
    oclblas_err_t finalize();
    oclblas_err_t createBinaryWithSource(const std::string &filename);
    oclblas_err_t buildProgramFromBinary(const std::string &program_name);
    oclblas_err_t createKernel(const std::string &kernel_name, std::string program_name="NULL");
    oclblas_err_t releaseKernel(const std::string &kernel_name);

    cl_platform_id getPlatform() { return oclblas_platform; }
    cl_uint getNumDevices() { return oclblas_num_devices; }
    cl_device_id *getDevices() { return oclblas_devices; }
    cl_context getContext() { return oclblas_context; }
    cl_command_queue *getCommandQueues() { return oclblas_command_queues; }
    cl_kernel getKernel(const std::string &kernel_name) { return kernel_pool[kernel_name]; }
    oclblas_err_t getPlatformInfo(const cl_platform_info param_name, void *param_value);
    oclblas_err_t getDeviceInfo(const cl_device_info param_name, void *param_value, const int device_idx=0);
    std::string getErrorMessage(const int error_value);
    unsigned int getPlatformNo() { return platform_no; }
    unsigned int getDeviceType() { return device_type; }
    cl_mem getBufA(const size_t required_bufa_size);
    cl_mem getBufB(const size_t required_bufb_size);
};

#endif // #ifndef OCLBLAS_RUNTIME_HPP_
