#include <stdlib.h>
#include "benchmark.h"

double oclblas_get_current_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}
