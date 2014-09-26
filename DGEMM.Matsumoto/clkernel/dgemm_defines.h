// dgemmK_CNT_48x64x16_3x4x1_16c16_16a16_4b64_vw1_shrAB_k2AB_nusM_skw_fma_rplB_cbcArbcB_ba
#define DGEMMK_BODY(KFIRST, KLAST) \
{ \
  __local double __a[48][16+1]; \
  __local double __b[64][16+1]; \
  const int __1dimId = (get_local_id(1)<<4) + get_local_id(0); \
  A += ((get_global_id(0)-get_local_id(0))*3)<<4; \
  const int __ia = __1dimId%16; \
  const int __ic = get_local_id(0); \
  const int __pa = (__1dimId>>4)<<0; \
  B += ((get_global_id(1)-get_local_id(1))<<2)*k; \
  const int __jb = (__1dimId%64)<<0; \
  const int __jc = get_local_id(1)<<2; \
  const int __pb = __1dimId>>6; \
  c[0][0] = (double)0.0; \
  c[0][1] = (double)0.0; \
  c[0][2] = (double)0.0; \
  c[1][0] = (double)0.0; \
  c[1][1] = (double)0.0; \
  c[1][2] = (double)0.0; \
  c[2][0] = (double)0.0; \
  c[2][1] = (double)0.0; \
  c[2][2] = (double)0.0; \
  c[3][0] = (double)0.0; \
  c[3][1] = (double)0.0; \
  c[3][2] = (double)0.0; \
  for (int l = KFIRST; l < KLAST; l += 16) { \
    __a[__ia+0][__pa+0] = A[l*(lda>>0)+((__pa+0)*48)+__ia+0]; \
    __a[__ia+16][__pa+0] = A[l*(lda>>0)+((__pa+0)*48)+__ia+16]; \
    __a[__ia+32][__pa+0] = A[l*(lda>>0)+((__pa+0)*48)+__ia+32]; \
    __b[__jb+0][__pb+0] = B[((l+__pb+0)<<6)+(__jb+0)]; \
    __b[__jb+0][__pb+4] = B[((l+__pb+4)<<6)+(__jb+0)]; \
    __b[__jb+0][__pb+8] = B[((l+__pb+8)<<6)+(__jb+0)]; \
    __b[__jb+0][__pb+12] = B[((l+__pb+12)<<6)+(__jb+0)]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int p = 0; p < 16; p += 1) { \
      double a[3], b[4]; \
      a[0] = __a[__ic+0][p+0]; \
      a[1] = __a[__ic+16][p+0]; \
      a[2] = __a[__ic+32][p+0]; \
      b[0] = __b[__jc+0][p+0]; \
      b[1] = __b[__jc+1][p+0]; \
      b[2] = __b[__jc+2][p+0]; \
      b[3] = __b[__jc+3][p+0]; \
      c[0][0] = fma(a[0], (double)b[0], c[0][0]); \
      c[0][1] = fma(a[1], (double)b[0], c[0][1]); \
      c[0][2] = fma(a[2], (double)b[0], c[0][2]); \
      c[1][0] = fma(a[0], (double)b[1], c[1][0]); \
      c[1][1] = fma(a[1], (double)b[1], c[1][1]); \
      c[1][2] = fma(a[2], (double)b[1], c[1][2]); \
      c[2][0] = fma(a[0], (double)b[2], c[2][0]); \
      c[2][1] = fma(a[1], (double)b[2], c[2][1]); \
      c[2][2] = fma(a[2], (double)b[2], c[2][2]); \
      c[3][0] = fma(a[0], (double)b[3], c[3][0]); \
      c[3][1] = fma(a[1], (double)b[3], c[3][1]); \
      c[3][2] = fma(a[2], (double)b[3], c[3][2]); \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
  } \
  C += offsetC; \
}
