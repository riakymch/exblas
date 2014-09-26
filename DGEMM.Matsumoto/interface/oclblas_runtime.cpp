#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "oclblas_runtime.hpp"

using std::ifstream;
using std::ios;
using std::map;
using std::string;
using std::vector;

oclblas_runtime::oclblas_runtime()
{
  initialized = false;
  creating_kernels = true;
  oclblas_kernel_dir = "NULL";
}

oclblas_runtime::oclblas_runtime(const bool _creating_kernels)
{
  initialized = false;
  creating_kernels = _creating_kernels;
  oclblas_kernel_dir = "NULL";
}

oclblas_runtime::oclblas_runtime(const bool _creating_kernels, const string _oclblas_kernel_dir)
{
  initialized = false;
  creating_kernels = _creating_kernels;
  if (_oclblas_kernel_dir[_oclblas_kernel_dir.length()-1] == '/') {
    oclblas_kernel_dir = _oclblas_kernel_dir;
  } else {
    oclblas_kernel_dir = _oclblas_kernel_dir + "/";
  }
}

oclblas_err_t
oclblas_runtime::createBinaryWithSource(const string &filename)
{
  if (!initialized) {
    fprintf(stderr, "ERROR [%d in %s]: oclblas has not been initialized\n", __LINE__, __FILE__);
    return OCLBLAS_RT_NOT_INITIALIZED;
  }

  const char *filename_cstr = (oclblas_kernel_dir + filename).c_str();
  const size_t filename_len = strlen(filename_cstr);
  char *binary_name = (char *)malloc(filename_len+5);
  const char *kernelname_end = strstr(filename_cstr+filename_len-3, ".cl");
  if (kernelname_end != NULL) {
    char *tmp_cstr = (char *)malloc(filename_len+1);
    strcpy(tmp_cstr, filename_cstr);
    tmp_cstr[strlen(filename_cstr)-strlen(kernelname_end)] = '\0';
    sprintf(binary_name, "%s.bin", tmp_cstr);
    free(tmp_cstr);
  } else {
    sprintf(binary_name, "%s.bin", filename_cstr);
  }

  /********************************************************************************
   * Read OpenCL kernel source
   ********************************************************************************/
  FILE *source_handle = fopen(filename_cstr, "r");

  if (source_handle == NULL) {
    fprintf(stderr, "ERROR [%d in %s]: Failed to open %s file\n", __LINE__, __FILE__, filename_cstr);
    return OCLBLAS_OPEN_FILE_FAILURE;
  }

  fseek(source_handle, 0, SEEK_END);
  size_t source_size = ftell(source_handle);
  rewind(source_handle);

  char *source_buffer = (char *)malloc(source_size + 1);
  source_buffer[source_size] = '\0';
  size_t read_size = fread(source_buffer, sizeof(char), source_size, source_handle);
  if (read_size < source_size) {
    fprintf(stderr, "ERROR [%d in %s]: Failed to read %s file\n", __LINE__, __FILE__, filename_cstr);
    return OCLBLAS_READ_FILE_FAILURE;
  }
  fclose(source_handle);

  /********************************************************************************
   * Create and build OpenCL program with source
   ********************************************************************************/
  cl_program program = clCreateProgramWithSource(oclblas_context, 1, (const char**)&source_buffer, NULL, &oclblas_err);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create program with source %s\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, filename_cstr);
    return oclblas_err;
  }
  free(source_buffer);

  //char compile_options[] = "-cl-denorms-are-zero -DFP_FAST_FMAF";
  char compile_options[] = "-I./";
  oclblas_err = clBuildProgram(program, 0, NULL, compile_options, NULL, NULL);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to build program with program %s\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, filename_cstr);
    fprintf(stderr, "################################################################################\n");
    fprintf(stderr, "# Program Build Log -- START                                                   #\n");
    fprintf(stderr, "################################################################################\n");
    size_t build_log_size;
    clGetProgramBuildInfo(program, oclblas_devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
    char *build_log = (char *)malloc(build_log_size + 1);
    build_log[build_log_size] = '\0';
    clGetProgramBuildInfo(program, oclblas_devices[0], CL_PROGRAM_BUILD_LOG, build_log_size+1, build_log, NULL);
    fprintf(stderr, "%s\n", build_log);
    fprintf(stderr, "################################################################################\n");
    fprintf(stderr, "# Program Build Log -- END                                                     #\n");
    fprintf(stderr, "################################################################################\n");
    free(build_log);
    return oclblas_err;
  }

  /********************************************************************************
   * Get OpenCL program binaries
   ********************************************************************************/
  cl_uint num_program_devices = 0;
  clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_program_devices, NULL);

  size_t *binary_sizes = (size_t *)malloc(num_program_devices*sizeof(size_t));
  oclblas_err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, num_program_devices*sizeof(size_t), binary_sizes, NULL);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to get program binary sizes\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }

  unsigned char **binaries = (unsigned char **)malloc(num_program_devices);
  for (unsigned int i = 0; i < num_program_devices; i++) {
    binaries[i] = (unsigned char *)malloc(binary_sizes[i]);
  }

  oclblas_err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, num_program_devices*sizeof(unsigned char *), binaries, NULL);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to get OpenCL program binaries\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }

  /********************************************************************************
   * Write OpenCL program binaries to file
   ********************************************************************************/
  FILE *binary_handle = fopen(binary_name, "wb");
  if (binary_handle == NULL) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to open file %s to write binaries\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, binary_name);
    return OCLBLAS_OPEN_FILE_FAILURE;
  }

  fwrite(&num_program_devices, sizeof(cl_uint), 1, binary_handle);
  fwrite(binary_sizes, sizeof(size_t), num_program_devices, binary_handle);
  for (unsigned int i = 0; i < num_program_devices; i++) {
    fwrite(binaries[i], 1, binary_sizes[i], binary_handle);
  }

  /********************************************************************************
   * Cleanup
   ********************************************************************************/
  for (unsigned int i = 0; i < num_program_devices; i++) {
    free(binaries[i]);
  }
  free(binaries);
  free(binary_sizes);
  free(binary_name);
  fclose(binary_handle);

  return OCLBLAS_SUCCESS;
} // End of createBinaryWithSource(const std::string filename)

oclblas_err_t
oclblas_runtime::buildProgramFromBinary(const std::string &program_name)
{
  const char *c_program_name = (oclblas_kernel_dir + program_name).c_str();
  char *filename_cstr = (char *)malloc(strlen(c_program_name)+5);
  sprintf(filename_cstr, "%s.bin", c_program_name);
  FILE *binary_handle = fopen(filename_cstr, "rb");
  if (binary_handle == NULL) {
    fprintf(stderr, "ERROR [%d in %s]: Failed to open file %s\n", __LINE__, __FILE__, filename_cstr);
    return OCLBLAS_OPEN_FILE_FAILURE;
  }

  /********************************************************************************
   * Read the number of OpenCL program devices
   ********************************************************************************/
  cl_uint num_program_devices;
  size_t ret  = fread(&num_program_devices, sizeof(cl_uint), 1, binary_handle);
  if (ret < 1) {
    fprintf(stderr, "ERROR [%d in %s]: Failed to read the number of OpenCL program devices for binary %s\n", __LINE__, __FILE__, filename_cstr);
    return OCLBLAS_READ_FILE_FAILURE;
  }
  if (num_program_devices < oclblas_num_devices) {
    fprintf(stderr, "ERROR [%d in %s]: Invalid number of program devices for binary %s (obtained #devices %u < designated #devices %u)\n", __LINE__, __FILE__, filename_cstr, num_program_devices, oclblas_num_devices);
    return OCLBLAS_INVALID_VALUE;
  }

  /********************************************************************************
   * Read the size of binaries
   ********************************************************************************/
  size_t *binary_sizes = (size_t *)malloc(num_program_devices*sizeof(size_t));
  ret = fread(binary_sizes, sizeof(size_t), num_program_devices, binary_handle);

  /********************************************************************************
   * Read binaries
   ********************************************************************************/
  unsigned char **binaries = (unsigned char **)malloc(num_program_devices*sizeof(unsigned char *));
  for (unsigned int i = 0; i < num_program_devices; i++) {
    binaries[i] = (unsigned char *)malloc(binary_sizes[i]);
    ret = fread(binaries[i], 1, binary_sizes[i], binary_handle);
    if (ret < binary_sizes[i]) {
      fprintf(stderr, "ERROR [%d in %s]: Failed to read binary %s for device %d\n", __LINE__, __FILE__, filename_cstr, i);
      return OCLBLAS_READ_FILE_FAILURE;
    }
  }
  fclose(binary_handle);

  /********************************************************************************
   * Create OpenCL program with binary
   ********************************************************************************/
  cl_int binary_status;
  cl_program program = clCreateProgramWithBinary(oclblas_context, num_program_devices, oclblas_devices, (const size_t *)binary_sizes, (const unsigned char **)binaries, &binary_status, &oclblas_err);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create OpenCL program %s with binary\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, c_program_name);
    return oclblas_err;
  }

  /********************************************************************************
   * Build OpenCL program and put the program in program_pool
   ********************************************************************************/
  oclblas_err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to build OpenCL program %s with binary\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, c_program_name);
    return oclblas_err;
  }

  program_pool[program_name] = program;

  /********************************************************************************
   * Cleanup
   ********************************************************************************/
  for (unsigned int i = 0; i < num_program_devices; i++) {
    free(binaries[i]);
  }
  free(binaries);
  free(binary_sizes);
  free(filename_cstr);

  return OCLBLAS_SUCCESS;
} // End of buildProgramFromBinary(const std::string program_name)

oclblas_err_t
oclblas_runtime::createKernel(const std::string &kernel_name, std::string program_name)
{
  if (!initialized) {
    fprintf(stderr, "ERROR [%d in %s]: oclblas has not been initialized\n", __LINE__, __FILE__);
    return OCLBLAS_RT_NOT_INITIALIZED;
  }

  if (program_name == "NULL") {
    program_name = kernel_name;
  }

  if (program_pool.find(program_name) == program_pool.end()) {
    //fprintf(stderr, "ERROR (%s) [%d in %s]: Couln't find %s OpenCL program for creating the kernel\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, program_name.c_str());
    //return OCLBLAS_PROGRAM_NOT_FIND;
    oclblas_err = buildProgramFromBinary(program_name);
    if(oclblas_err != OCLBLAS_SUCCESS)
      return oclblas_err;
  }
  cl_program program = program_pool[program_name];

  kernel_pool[kernel_name] = clCreateKernel(program, kernel_name.c_str(), &oclblas_err);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create OpenCL kenrel %s\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, kernel_name.c_str());
    return oclblas_err;
  }

  return OCLBLAS_SUCCESS;
}

oclblas_err_t
oclblas_runtime::releaseKernel(const std::string &kernel_name)
{
  oclblas_err = clReleaseKernel(kernel_pool[kernel_name]);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to release OpenCL kenrel %s\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, kernel_name.c_str());
    return oclblas_err;
  }
  kernel_pool.erase(kernel_name);
  return OCLBLAS_SUCCESS;
}

oclblas_err_t
oclblas_runtime::initialize(const cl_platform_id &platform, const cl_uint num_devices, cl_device_id *devices, const cl_context &context)
{
  if (initialized) {
    fprintf(stderr, "WARNING [%d in %s]: oclblas runtime has already been initialized\n", __LINE__, __FILE__);
    return OCLBLAS_RT_ALREADY_INITIALIZED;
  }

  oclblas_platform    = platform;
  oclblas_num_devices = num_devices;
  oclblas_devices     = devices;
  oclblas_context     = context;

  /********************************************************************************
   * Create OpenCL Command Queues
   ********************************************************************************/
  // Discussion: Creating a single queue is enough?
  oclblas_command_queues = (cl_command_queue *)malloc(num_devices*sizeof(cl_command_queue));
  for (unsigned int i = 0; i < num_devices; i++) {
    oclblas_command_queues[i] = clCreateCommandQueue(oclblas_context, oclblas_devices[i], CL_QUEUE_PROFILING_ENABLE, &oclblas_err);
    if (oclblas_err != CL_SUCCESS) {
      fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create OpenCL command queue for device %d\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, i);
      return oclblas_err;
    }
  }

  /********************************************************************************
   * Determine the difectory for OpenCL kernel sources and binaries
   ********************************************************************************/
  if (oclblas_kernel_dir == "NULL") {
    char *kernel_dir_cstr = getenv("OCLBLAS_KERNEL_DIR");
    if (kernel_dir_cstr != NULL) {
      oclblas_kernel_dir = string(kernel_dir_cstr) + "/";
    } else {
      oclblas_kernel_dir = "../clkernel/";
    }
  }

  bufa_size = bufb_size = 0;
  bufa = bufb = NULL;

  initialized = true;

  /********************************************************************************
   * Create OpenCL Kernels
   ********************************************************************************/
  if (creating_kernels) {
    // Level-1 BLAS
    if ((oclblas_err = createKernel("dcopy",              "dcopy" )) != OCLBLAS_SUCCESS) return oclblas_err;
    // Level-3 BLAS
    //if ((oclblas_err = createKernel("dgemm",                         "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("dgemm_alpha_zero",              "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("dgemmK",                        "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("dgemmK_copyA",                  "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("dgemmK_copyA_trans",            "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("dgemmK_copyB",                  "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("dgemmK_copyB_trans",            "dgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsymm",                         "dsymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsymmK_copy_left_upper",        "dsymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsymmK_copy_left_lower",        "dsymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsymmK_copy_right_upper",       "dsymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsymmK_copy_right_lower",       "dsymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsyrk0_lower",                  "dsyrk0")) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsyrkK_upper",                  "dsyrk" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsyrkK_lower",                  "dsyrk" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dsyr2k",                        "dsyr2k")) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmm",                         "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_upper",                  "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_lower",                  "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmm0_left_upper",             "dtrmm0")) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_left_upper",        "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_left_upper_trans",  "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_left_lower",        "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_left_lower_trans",  "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_right_upper",       "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_right_upper_trans", "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_right_lower",       "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrmmK_copy_right_lower_trans", "dtrmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("dtrsm",                         "dtrsm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("sgemm",                         "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("sgemm_alpha_zero",              "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("sgemmK",                        "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("sgemmK_copyA",                  "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("sgemmK_copyA_trans",            "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("sgemmK_copyB",                  "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    if ((oclblas_err = createKernel("sgemmK_copyB_trans",            "sgemm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssymm",                         "ssymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssymmK_copy_left_upper",        "ssymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssymmK_copy_left_lower",        "ssymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssymmK_copy_right_upper",       "ssymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssymmK_copy_right_lower",       "ssymm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssyrk",                         "ssyrk" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssyrkK_upper",                  "ssyrk" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssyrkK_lower",                  "ssyrk" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("ssyr2k",                        "ssyr2k")) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmm",                         "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_upper",                  "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_lower",                  "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_left_upper",        "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_left_upper_trans",  "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_left_lower",        "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_left_lower_trans",  "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_right_upper",       "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_right_upper_trans", "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_right_lower",       "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strmmK_copy_right_lower_trans", "strmm" )) != OCLBLAS_SUCCESS) return oclblas_err;
    //if ((oclblas_err = createKernel("strsm",                         "strsm" )) != OCLBLAS_SUCCESS) return oclblas_err;
  }

  return OCLBLAS_SUCCESS;
}

oclblas_err_t
oclblas_runtime::initialize(const cl_platform_id &platform, const cl_uint num_devices, cl_device_id *devices)
{
  if (initialized) {
    fprintf(stderr, "WARNING (%s) [%d in %s]: oclblas runtime has already been initialized\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return OCLBLAS_RT_ALREADY_INITIALIZED;
  }

  /********************************************************************************
   * Create OpenCL Context
   ********************************************************************************/
  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &oclblas_err);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create OpenCL context\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }

  return initialize(platform, num_devices, devices, context);
}

oclblas_err_t
oclblas_runtime::initialize(const unsigned int target_platform_no, const unsigned int target_device_type)
{
  platform_no = target_platform_no;
  device_type = target_device_type;

  /********************************************************************************
   * Create OpenCL Platforms
   ********************************************************************************/
  cl_uint num_platforms;
  oclblas_err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to get the number of OpenCL platforms\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }
  if (target_platform_no >= num_platforms) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: No such platform %u (>= %u)\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__, target_platform_no, num_platforms);
    return OCLBLAS_INVALID_PLATFORM;
  }

  cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms*sizeof(cl_platform_id));
  oclblas_err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to get OpenCL platforms\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }
  cl_platform_id platform = platforms[target_platform_no];

  /********************************************************************************
   * Create OpenCL Devices
   ********************************************************************************/
  cl_uint num_devices;
  oclblas_err = clGetDeviceIDs(platform, target_device_type, 0, NULL, &num_devices);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to get the number of OpenCL devices\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }

  cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
  oclblas_err = clGetDeviceIDs(platform, target_device_type, num_devices, devices, NULL);
  if (oclblas_err != CL_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to get OpenCL devices\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
    return oclblas_err;
  }

  free(platforms);
  return initialize(platform, num_devices, devices);
}

oclblas_err_t
oclblas_runtime::initialize()
{
  return initialize(OCLBLAS_DEFAULT_PLATFORM, OCLBLAS_DEFAULT_DEVICE_TYPE);
}

oclblas_err_t
oclblas_runtime::finalize()
{
  if (!initialized) {
    //fprintf(stderr, "WARNING [%d in %s]: oclblas has not been initialized\n", __LINE__, __FILE__);
    return OCLBLAS_RT_NOT_INITIALIZED;
  }

  for (map<string,cl_kernel>::iterator itr = kernel_pool.begin(); itr != kernel_pool.end(); itr++) {
    clReleaseKernel(itr->second);
  }

  if (oclblas_devices) free(oclblas_devices);
  if (oclblas_context) clReleaseContext(oclblas_context);
  if (oclblas_command_queues) free(oclblas_command_queues);

  oclblas_platform = NULL;
  oclblas_num_devices = 0;
  oclblas_devices = NULL;
  oclblas_context = NULL;
  oclblas_command_queues = NULL;

  clReleaseMemObject(bufa);
  clReleaseMemObject(bufb);
  bufa = bufb = NULL;
  bufa_size = bufb_size = 0;

  initialized = false;
  return OCLBLAS_SUCCESS;
}

oclblas_err_t
oclblas_runtime::getPlatformInfo(const cl_platform_info param_name, void *param_value)
{
  switch (param_name) {
    case CL_PLATFORM_PROFILE:
    case CL_PLATFORM_VERSION:
    case CL_PLATFORM_NAME:
    case CL_PLATFORM_VENDOR:
    case CL_PLATFORM_EXTENSIONS:
      return clGetPlatformInfo(oclblas_platform, param_name, MAX_INFO_PARAM_SIZE, param_value, NULL);
      break;
  }
  return OCLBLAS_INVALID_VALUE;
}

oclblas_err_t
oclblas_runtime::getDeviceInfo(const cl_device_info param_name, void *param_value, const int device_idx)
{
  const cl_device_id device = oclblas_devices[device_idx];
  switch (param_name) {
    case CL_DEVICE_ADDRESS_BITS:
    case CL_DEVICE_MAX_CLOCK_FREQUENCY:
    case CL_DEVICE_MAX_COMPUTE_UNITS:
    case CL_DEVICE_MAX_CONSTANT_ARGS:
    case CL_DEVICE_MAX_READ_IMAGE_ARGS:
    case CL_DEVICE_MAX_SAMPLERS:
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
    case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
    case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
    case CL_DEVICE_VENDOR_ID:
#ifdef CL_VERSION_1_2
    case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
    case CL_DEVICE_REFERENCE_COUNT:
#endif
      return clGetDeviceInfo(device, param_name, sizeof(cl_uint), param_value, NULL);
      break;
    case CL_DEVICE_AVAILABLE:
    case CL_DEVICE_COMPILER_AVAILABLE:
    case CL_DEVICE_ENDIAN_LITTLE:
    case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
    case CL_DEVICE_HOST_UNIFIED_MEMORY:
    case CL_DEVICE_IMAGE_SUPPORT:
#ifdef CL_VERSION_1_2
    case CL_DEVICE_LINKER_AVAILABLE:
    case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
#endif
      return clGetDeviceInfo(device, param_name, sizeof(cl_bool), param_value, NULL);
      break;
    case CL_DEVICE_EXTENSIONS:
    case CL_DEVICE_NAME:
    case CL_DEVICE_OPENCL_C_VERSION:
    case CL_DEVICE_PROFILE:
    case CL_DEVICE_VENDOR:
    case CL_DEVICE_VERSION:
    case CL_DRIVER_VERSION:
#ifdef CL_VERSION_1_2
    case CL_DEVICE_BUILT_IN_KERNELS:
#endif
      return clGetDeviceInfo(device, param_name, MAX_INFO_PARAM_SIZE*sizeof(char), param_value, NULL);
      break;
    //case CL_DEVICE_DOUBLE_FP_CONFIG:
    //case CL_DEVICE_HALF_FP_CONFIG:
    case CL_DEVICE_SINGLE_FP_CONFIG:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_fp_config), param_value, NULL);
      break;
    case CL_DEVICE_EXECUTION_CAPABILITIES:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_exec_capabilities), param_value, NULL);
      break;
    case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
    case CL_DEVICE_GLOBAL_MEM_SIZE:
    case CL_DEVICE_LOCAL_MEM_SIZE:
    case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
      return clGetDeviceInfo(device, param_name, sizeof(cl_ulong), param_value, NULL);
      break;
    case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_mem_cache_type), param_value, NULL);
      break;
    case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
      return clGetDeviceInfo(device, param_name, sizeof(cl_int), param_value, NULL);
      break;
    case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
    case CL_DEVICE_IMAGE2D_MAX_WIDTH:
    case CL_DEVICE_IMAGE3D_MAX_DEPTH:
    case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
    case CL_DEVICE_IMAGE3D_MAX_WIDTH:
    case CL_DEVICE_MAX_PARAMETER_SIZE:
    case CL_DEVICE_MAX_WORK_GROUP_SIZE:
    case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
#ifdef CL_VERSION_1_2
    case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
    case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
    case CL_DEVICE_PRINTF_BUFFER_SIZE:
#endif
      return clGetDeviceInfo(device, param_name, sizeof(size_t), param_value, NULL);
      break;
    case CL_DEVICE_LOCAL_MEM_TYPE:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_local_mem_type), param_value, NULL);
      break;
    case CL_DEVICE_MAX_WORK_ITEM_SIZES:
      return clGetDeviceInfo(device, param_name, 3*sizeof(size_t), param_value, NULL);
      break;
#ifdef CL_VERSION_1_2
    case CL_DEVICE_PARENT_DEVICE:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_id), param_value, NULL);
      break;
#endif
#ifdef CL_VERSION_1_2
    case CL_DEVICE_PARTITION_PROPERTIES:
    case CL_DEVICE_PARTITION_TYPE:
      return clGetDeviceInfo(device, param_name, 3*sizeof(cl_device_partition_property), param_value, NULL);
      break;
#endif
#ifdef CL_VERSION_1_2
    case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_affinity_domain), param_value, NULL);
      break;
#endif
    case CL_DEVICE_PLATFORM:
      return clGetDeviceInfo(device, param_name, sizeof(cl_platform_id), param_value, NULL);
      break;
    case CL_DEVICE_QUEUE_PROPERTIES:
      return clGetDeviceInfo(device, param_name, sizeof(cl_command_queue_properties), param_value, NULL);
      break;
    case CL_DEVICE_TYPE:
      return clGetDeviceInfo(device, param_name, sizeof(cl_device_type), param_value, NULL);
      break;
  }
  return OCLBLAS_INVALID_VALUE;
}

string
oclblas_runtime::getErrorMessage(const int error_value)
{
  switch (error_value) {
    case OCLBLAS_SUCCESS:
      return "No error"; break;
    case OCLBLAS_DEVICE_NOT_FOUND:
      return "Device not found"; break;
    case OCLBLAS_DEVICE_NOT_AVAILABLE:
      return "Device not available"; break;
    case OCLBLAS_COMPILER_NOT_AVAILABLE:
      return "Compliler not available"; break;
    case OCLBLAS_OUT_OF_HOST_MEMORY:
      return "Out of host memory"; break;
    case OCLBLAS_BUILD_PROGRAM_FAILURE:
      return "Build program failure"; break;
    case OCLBLAS_INVALID_VALUE:
      return "Invalid value"; break;
    case OCLBLAS_INVALID_DEVICE_TYPE:
      return "Invalid device type"; break;
    case OCLBLAS_INVALID_PLATFORM:
      return "Invalid platform"; break;
    case OCLBLAS_INVALID_DEVICE:
      return "Invalid device"; break;
    case OCLBLAS_INVALID_CONTEXT:
      return "Invalid context"; break;
    case OCLBLAS_INVALID_QUEUE_PROPERTIES:
      return "Invalid queue properties"; break;
    case OCLBLAS_INVALID_MEM_OBJECT:
      return "Invalid memory object"; break;
    case OCLBLAS_INVALID_BINARY:
      return "Invalid binary"; break;
    case OCLBLAS_INVALID_BUILD_OPTIONS:
      return "Invalid build options"; break;
    case OCLBLAS_INVALID_PROGRAM:
      return "Invalid program"; break;
    case OCLBLAS_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid program executable"; break;
    case OCLBLAS_INVALID_KERNEL_NAME:
      return "Invalid kernel name"; break;
    case OCLBLAS_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition"; break;
    case OCLBLAS_INVALID_KERNEL:
      return "Invalid kernel"; break;
    case OCLBLAS_INVALID_ARG_INDEX:
      return "Invalid argument index"; break;
    case OCLBLAS_INVALID_ARG_VALUE:
      return "Invalid argument value"; break;
    case OCLBLAS_INVALID_OPERATION:
      return "Invalid operation"; break;
    case OCLBLAS_INVALID_BUFFER_SIZE:
      return "Invalid buffer size"; break;
    case OCLBLAS_RT_ALREADY_INITIALIZED:
      return "Runtime already initialized"; break;
    case OCLBLAS_RT_NOT_INITIALIZED:
      return "Runtine not initialized"; break;
    case OCLBLAS_OPEN_FILE_FAILURE:
      return "Failed to open file"; break;
    case OCLBLAS_READ_FILE_FAILURE:
      return "Failed to read file"; break;
    case OCLBLAS_INVALID_FUNCTION_CALL:
      return "Invalid function call"; break;
    case OCLBLAS_PROGRAM_NOT_FIND:
      return "Program not found"; break;
    case OCLBLAS_INVALID_FUNC_ARGUMENT:
      return "Invalid function argument"; break;
    default:
      return "Unknown error"; break;
  }
}

cl_mem oclblas_runtime::getBufA(const size_t required_bufa_size)
{
  if (required_bufa_size > bufa_size) {
    bufa_size = required_bufa_size;
    clReleaseMemObject(bufa);
    bufa = clCreateBuffer(oclblas_context, CL_MEM_READ_WRITE, bufa_size, NULL, &oclblas_err);
    if (oclblas_err != CL_SUCCESS) {
      fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create temporary buffers\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
      return NULL;
    }
  }
  return bufa;
}

cl_mem oclblas_runtime::getBufB(const size_t required_bufb_size)
{
  if (required_bufb_size > bufb_size) {
    bufb_size = required_bufb_size;
    clReleaseMemObject(bufb);
    bufb = clCreateBuffer(oclblas_context, CL_MEM_READ_WRITE, bufb_size, NULL, &oclblas_err);
    if (oclblas_err != CL_SUCCESS) {
      fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to create temporary buffers\n", getErrorMessage(oclblas_err).c_str(), __LINE__, __FILE__);
      return NULL;
    }
  }
  return bufb;
}
