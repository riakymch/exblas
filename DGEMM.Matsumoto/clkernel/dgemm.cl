#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// When alpha equals zero.
  __kernel __attribute__((vec_type_hint(double)))
void dgemm_alpha_zero(
    const double beta,
    __global double *c, const int offsetc, const int ldc
    )
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  c[offsetc + i + j*ldc] *= beta;
}

#include "dgemm_defines.h"

// dgemmK_CNT_48x64x16_3x4x1_16c16_16a16_4b64_vw1_shrAB_k2AB_nusM_skw_fma_rplB_cbcArbcB_ba
  __kernel __attribute__((vec_type_hint(double))) __attribute__((work_group_size_hint(16, 16, 1)))
void dgemmK(
    const int m,
    const int n,
    const int k,
    __global double const * restrict A,
    __global double const * restrict B,
    __global double *C,
    const int lda,
    const int ldb,
    const int ldc,
    const int offsetC,
    const double alpha,
    const double beta
    )
{
  const int i = ((get_global_id(0)-get_local_id(0))*3) + (get_local_id(0)<<0);
  const int j = get_global_id(1)<<2;
  double c[4][3];
  DGEMMK_BODY(0, k);
  for (int j1 = 0; j1 < 4; j1++) {
    const int j2 = j+j1;
    for (int i1 = 0; i1 < 3; i1++) {
      const int i2 = i+(i1<<4);
      if (i2+0 < m && j2 < n)
        C[j2*ldc + i2+0] = fma(alpha, c[j1][i1], beta*C[j2*ldc + i2+0]);
    }
  }
}

// dgemmK_copy_CAN_cbc_48b16_24g2_1i2_1u2
__kernel void dgemmK_copyA(
    const int m,
    const int k,
    const int lda,
    const int pmk,
    const int offsetA,
    __global double const * restrict rA,
    __global double * restrict wA
    )
{
  const int I = (get_global_id(0)/48)*48;
  const int i = (get_global_id(0)<<0) - I;
  const int P = (get_global_id(1)>>3)<<4;
  const int p = (get_global_id(1)<<1) - P;
#pragma unroll 1
  for (int i1 = i; i1 < i+1; i1++) {
#pragma unroll 2
    for (int p1 = p; p1 < p+2; p1++) {
      wA[P*pmk+(I<<4)+(p1*48)+i1] = (I+i1 < m && P+p1 < k) ? rA[offsetA+(P+p1)*lda+I+i1] : 0.;
    }
  }
}

// dgemmK_copy_CAT_cbc_48b16_16g16_1i1_1u1
__kernel void dgemmK_copyA_trans(
    const int m,
    const int k,
    const int lda,
    const int pmk,
    const int offsetA,
    __global double const * restrict rA,
    __global double * restrict wA
    )
{
  const int I = (get_global_id(1)/48)*48;
  const int i = (get_global_id(1)<<0) - I;
  const int P = (get_global_id(0)>>4)<<4;
  const int p = (get_global_id(0)<<0) - P;
#pragma unroll 1
  for (int i1 = i; i1 < i+1; i1++) {
#pragma unroll 1
    for (int p1 = p; p1 < p+1; p1++) {
      wA[P*pmk+(I<<4)+(p1*48)+i1] = (I+i1 < m && P+p1 < k) ? rA[offsetA+(I+i1)*lda+P+p1] : 0.;
    }
  }
}

// dgemmK_copy_CBN_rbc_64b16_32g8_1i8_1u8
__kernel void dgemmK_copyB(
    const int n,
    const int k,
    const int ldb,
    const int pnk,
    const int offsetB,
    __global double const * restrict rB,
    __global double * restrict wB
    )
{
  const int J = (get_global_id(0)>>6)<<6;
  const int j = (get_global_id(0)<<0) - J;
  const int p = get_global_id(1)<<3;
#pragma unroll 1
  for (int j1 = j; j1 < j+1; j1++) {
#pragma unroll 8
    for (int p1 = p; p1 < p+8; p1++) {
      wB[(p1<<6)+J*pnk+j1] = (J+j1 < n && p1 < k) ? rB[offsetB+p1*ldb+J+j1] : 0.;
    }
  }
}

// dgemmK_copy_CBT_rbc_64b16_64g4_1i1_1u1
__kernel void dgemmK_copyB_trans(
    const int n,
    const int k,
    const int ldb,
    const int pnk,
    const int offsetB,
    __global double const * restrict rB,
    __global double * restrict wB
    )
{
  const int J = (get_global_id(1)>>6)<<6;
  const int j = (get_global_id(1)<<0) - J;
  const int p = get_global_id(0)<<0;
#pragma unroll 1
  for (int j1 = j; j1 < j+1; j1++) {
#pragma unroll 1
    for (int p1 = p; p1 < p+1; p1++) {
      wB[(p1<<6)+J*pnk+j1] = (J+j1 < n && p1 < k) ? rB[offsetB+(J+j1)*ldb+p1] : 0.;
    }
  }
}
