// When alpha equals zero.
  __kernel __attribute__((vec_type_hint(float)))
void sgemm_alpha_zero(
    const float beta,
    __global float *c, const int offsetc, const int ldc
    )
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  c[offsetc + i + j*ldc] *= beta;
}

#include "sgemm_defines.h"

// sgemmK_CNT_64x128x16_4x8x2_16c16_16a16_4b64_vw2_shrAB_k2AB_nusM_skw_mad_rplB_cbcArbcB_ba
  __kernel __attribute__((vec_type_hint(float2))) __attribute__((work_group_size_hint(16, 16, 1)))
void sgemmK(
    const int m,
    const int n,
    const int k,
    __global float2 const * restrict A,
    __global float2 const * restrict B,
    __global float *C,
    const int lda,
    const int ldb,
    const int ldc,
    const int offsetC,
    const float alpha,
    const float beta
    )
{
  const int i = ((get_global_id(0)-get_local_id(0))<<2) + (get_local_id(0)<<1);
  const int j = get_global_id(1)<<3;
  float2 c[8][2];
  SGEMMK_BODY(0, k);
  for (int j1 = 0; j1 < 8; j1++) {
    const int j2 = j+j1;
    for (int i1 = 0; i1 < 2; i1++) {
      const int i2 = i+(i1<<5);
      if (i2+0 < m && j2 < n)
        C[j2*ldc + i2+0] = mad(alpha, c[j1][i1].s0, beta*C[j2*ldc + i2+0]);
      if (i2+1 < m && j2 < n)
        C[j2*ldc + i2+1] = mad(alpha, c[j1][i1].s1, beta*C[j2*ldc + i2+1]);
    }
  }
}

// sgemmK_copy_CAN_cbc_64b16_32g8_1i8_1u2
__kernel void sgemmK_copyA(
    const int m,
    const int k,
    const int lda,
    const int pmk,
    const int offsetA,
    __global float const * restrict rA,
    __global float * restrict wA
    )
{
  const int I = (get_global_id(0)>>6)<<6;
  const int i = (get_global_id(0)<<0) - I;
  const int P = (get_global_id(1)>>1)<<4;
  const int p = (get_global_id(1)<<3) - P;
#pragma unroll 1
  for (int i1 = i; i1 < i+1; i1++) {
#pragma unroll 2
    for (int p1 = p; p1 < p+8; p1++) {
      wA[P*pmk+(I<<4)+(p1<<6)+i1] = (I+i1 < m && P+p1 < k) ? rA[offsetA+(P+p1)*lda+I+i1] : 0.f;
    }
  }
}

// sgemmK_copy_CAT_cbc_64b16_16g8_1i1_1u1
__kernel void sgemmK_copyA_trans(
    const int m,
    const int k,
    const int lda,
    const int pmk,
    const int offsetA,
    __global float const * restrict rA,
    __global float * restrict wA
    )
{
  const int I = (get_global_id(1)>>6)<<6;
  const int i = (get_global_id(1)<<0) - I;
  const int P = (get_global_id(0)>>4)<<4;
  const int p = (get_global_id(0)<<0) - P;
#pragma unroll 1
  for (int i1 = i; i1 < i+1; i1++) {
#pragma unroll 1
    for (int p1 = p; p1 < p+1; p1++) {
      wA[P*pmk+(I<<4)+(p1<<6)+i1] = (I+i1 < m && P+p1 < k) ? rA[offsetA+(I+i1)*lda+P+p1] : 0.f;
    }
  }
}

// sgemmK_copy_CBN_rbc_128b16_128g4_1i4_1u2
__kernel void sgemmK_copyB(
    const int n,
    const int k,
    const int ldb,
    const int pnk,
    const int offsetB,
    __global float const * restrict rB,
    __global float * restrict wB
    )
{
  const int J = (get_global_id(0)>>7)<<7;
  const int j = (get_global_id(0)<<0) - J;
  const int p = get_global_id(1)<<2;
#pragma unroll 1
  for (int j1 = j; j1 < j+1; j1++) {
#pragma unroll 2
    for (int p1 = p; p1 < p+4; p1++) {
      wB[(p1<<7)+J*pnk+j1] = (J+j1 < n && p1 < k) ? rB[offsetB+p1*ldb+J+j1] : 0.f;
    }
  }
}

// sgemmK_copy_CBT_rbc_128b16_32g8_1i1_1u1
__kernel void sgemmK_copyB_trans(
    const int n,
    const int k,
    const int ldb,
    const int pnk,
    const int offsetB,
    __global float const * restrict rB,
    __global float * restrict wB
    )
{
  const int J = (get_global_id(1)>>7)<<7;
  const int j = (get_global_id(1)<<0) - J;
  const int p = get_global_id(0)<<0;
#pragma unroll 1
  for (int j1 = j; j1 < j+1; j1++) {
#pragma unroll 1
    for (int p1 = p; p1 < p+1; p1++) {
      wB[(p1<<7)+J*pnk+j1] = (J+j1 < n && p1 < k) ? rB[offsetB+(J+j1)*ldb+p1] : 0.f;
    }
  }
}
