// sgemmK_CNT_64x128x16_4x8x2_16c16_16a16_4b64_vw2_shrAB_k2AB_nusM_skw_mad_rplB_cbcArbcB_ba
#define SGEMMK_BODY(KFIRST, KLAST) \
{ \
  __local float2 __a[32][16+1]; \
  __local float2 __b[64][16+1]; \
  const int __1dimId = (get_local_id(1)<<4) + get_local_id(0); \
  A += ((get_global_id(0)-get_local_id(0))<<1)<<4; \
  const int __ia = __1dimId%16; \
  const int __ic = get_local_id(0); \
  const int __pa = (__1dimId>>4)<<0; \
  B += ((get_global_id(1)-get_local_id(1))<<2)*k; \
  const int __jb = (__1dimId%64)<<0; \
  const int __jc = get_local_id(1)<<2; \
  const int __pb = __1dimId>>6; \
  c[0][0] = (float2)0.0f; \
  c[0][1] = (float2)0.0f; \
  c[1][0] = (float2)0.0f; \
  c[1][1] = (float2)0.0f; \
  c[2][0] = (float2)0.0f; \
  c[2][1] = (float2)0.0f; \
  c[3][0] = (float2)0.0f; \
  c[3][1] = (float2)0.0f; \
  c[4][0] = (float2)0.0f; \
  c[4][1] = (float2)0.0f; \
  c[5][0] = (float2)0.0f; \
  c[5][1] = (float2)0.0f; \
  c[6][0] = (float2)0.0f; \
  c[6][1] = (float2)0.0f; \
  c[7][0] = (float2)0.0f; \
  c[7][1] = (float2)0.0f; \
  for (int l = KFIRST; l < KLAST; l += 16) { \
    __a[__ia+0][__pa+0] = A[l*(lda>>1)+((__pa+0)<<5)+__ia+0]; \
    __a[__ia+16][__pa+0] = A[l*(lda>>1)+((__pa+0)<<5)+__ia+16]; \
    __b[__jb+0][__pb+0] = B[((l+__pb+0)<<6)+(__jb+0)]; \
    __b[__jb+0][__pb+4] = B[((l+__pb+4)<<6)+(__jb+0)]; \
    __b[__jb+0][__pb+8] = B[((l+__pb+8)<<6)+(__jb+0)]; \
    __b[__jb+0][__pb+12] = B[((l+__pb+12)<<6)+(__jb+0)]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int p = 0; p < 16; p += 2) { \
      float2 a[2], b[4]; \
      a[0] = __a[__ic+0][p+0]; \
      a[1] = __a[__ic+16][p+0]; \
      b[0] = __b[__jc+0][p+0]; \
      b[1] = __b[__jc+1][p+0]; \
      b[2] = __b[__jc+2][p+0]; \
      b[3] = __b[__jc+3][p+0]; \
      c[0][0] = mad(a[0], (float2)b[0].s0, c[0][0]); \
      c[0][1] = mad(a[1], (float2)b[0].s0, c[0][1]); \
      c[1][0] = mad(a[0], (float2)b[0].s1, c[1][0]); \
      c[1][1] = mad(a[1], (float2)b[0].s1, c[1][1]); \
      c[2][0] = mad(a[0], (float2)b[1].s0, c[2][0]); \
      c[2][1] = mad(a[1], (float2)b[1].s0, c[2][1]); \
      c[3][0] = mad(a[0], (float2)b[1].s1, c[3][0]); \
      c[3][1] = mad(a[1], (float2)b[1].s1, c[3][1]); \
      c[4][0] = mad(a[0], (float2)b[2].s0, c[4][0]); \
      c[4][1] = mad(a[1], (float2)b[2].s0, c[4][1]); \
      c[5][0] = mad(a[0], (float2)b[2].s1, c[5][0]); \
      c[5][1] = mad(a[1], (float2)b[2].s1, c[5][1]); \
      c[6][0] = mad(a[0], (float2)b[3].s0, c[6][0]); \
      c[6][1] = mad(a[1], (float2)b[3].s0, c[6][1]); \
      c[7][0] = mad(a[0], (float2)b[3].s1, c[7][0]); \
      c[7][1] = mad(a[1], (float2)b[3].s1, c[7][1]); \
      a[0] = __a[__ic+0][p+1]; \
      a[1] = __a[__ic+16][p+1]; \
      b[0] = __b[__jc+0][p+1]; \
      b[1] = __b[__jc+1][p+1]; \
      b[2] = __b[__jc+2][p+1]; \
      b[3] = __b[__jc+3][p+1]; \
      c[0][0] = mad(a[0], (float2)b[0].s0, c[0][0]); \
      c[0][1] = mad(a[1], (float2)b[0].s0, c[0][1]); \
      c[1][0] = mad(a[0], (float2)b[0].s1, c[1][0]); \
      c[1][1] = mad(a[1], (float2)b[0].s1, c[1][1]); \
      c[2][0] = mad(a[0], (float2)b[1].s0, c[2][0]); \
      c[2][1] = mad(a[1], (float2)b[1].s0, c[2][1]); \
      c[3][0] = mad(a[0], (float2)b[1].s1, c[3][0]); \
      c[3][1] = mad(a[1], (float2)b[1].s1, c[3][1]); \
      c[4][0] = mad(a[0], (float2)b[2].s0, c[4][0]); \
      c[4][1] = mad(a[1], (float2)b[2].s0, c[4][1]); \
      c[5][0] = mad(a[0], (float2)b[2].s1, c[5][0]); \
      c[5][1] = mad(a[1], (float2)b[2].s1, c[5][1]); \
      c[6][0] = mad(a[0], (float2)b[3].s0, c[6][0]); \
      c[6][1] = mad(a[1], (float2)b[3].s0, c[6][1]); \
      c[7][0] = mad(a[0], (float2)b[3].s1, c[7][0]); \
      c[7][1] = mad(a[1], (float2)b[3].s1, c[7][1]); \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
  } \
  C += offsetC; \
}
