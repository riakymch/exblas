#include <cstdio>
#include <cstring>
#include <string>
#include <CL/cl.h>
#include "oclblas_runtime.hpp"

using std::string;

int main(int argc, char *argv[])
{
  if (argc <= 1) {
    printf("Usage:\n");
    printf("1. cl2bin [-P platform_no] [-D device_type] [-R directory_name] opencl_kernel_source_filename\n");
    printf("2. ...\n");
    return 1;
  }

  unsigned int platform_no = OCLBLAS_DEFAULT_PLATFORM;
  unsigned int device_type = OCLBLAS_DEFAULT_DEVICE_TYPE;
  string dirname("");
  int i = 1;
  for ( ; i < argc-1; i++) {
    if (argv[i][0] == '-') {
      switch (toupper(argv[i][1])) {
        case 'D': // Device type
          if (++i >= argc) return 2;
          if (strstr(argv[i], "CPU") != NULL) {
            device_type = CL_DEVICE_TYPE_CPU;
          } else if (strstr(argv[i], "GPU") != NULL) {
            device_type = CL_DEVICE_TYPE_GPU;
          } else if (strstr(argv[i], "ACCELERATOR") != NULL) {
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
          } else if (strstr(argv[i], "DEFAULT") != NULL) {
            device_type = CL_DEVICE_TYPE_DEFAULT;
          } else {
            return 2;
          }
          break;
        case 'P': // Platform number
          if (++i >= argc) return 3;
          platform_no = atoi(argv[i]);
          break;
        case 'R':
          if (++i >= argc) return 4;
          dirname = argv[i];
          break;
      }
    }
  }
  string filename(argv[i]);

  oclblas_runtime rt(false, dirname);
  if (dirname == "") {
    rt = oclblas_runtime(false);
  } else {
    rt = oclblas_runtime(false, dirname);
  }

  oclblas_err_t err;
  err = rt.initialize(platform_no, device_type);
  if (err != OCLBLAS_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to initizlize oclblas_runtime\n", oclblas_get_error_message(err), __LINE__, __FILE__);
    return 1;
  }
  err = rt.createBinaryWithSource(filename);
  if (err != OCLBLAS_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to compile OpenCL kernel source %s\n", oclblas_get_error_message(err), __LINE__, __FILE__, filename.c_str());
    return 1;
  }
  string program_name = filename.erase(filename.find(".cl"));
  err = rt.buildProgramFromBinary(program_name.c_str());
  if (err != OCLBLAS_SUCCESS) {
    fprintf(stderr, "ERROR (%s) [%d in %s]: Failed to build OpenCL program from binary for %s\n", oclblas_get_error_message(err), __LINE__, __FILE__, program_name.c_str());
    return 1;
  }

  return 0;
}
