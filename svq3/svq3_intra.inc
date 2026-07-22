// Intra prediction + check_intra helpers, from ffmpeg release/8.1 libavcodec/h264pred.c + h264pred_template.c (BIT_DEPTH=8) + h264_parse.c (ff_h264_check_intra*_pred_mode)

// BIT_DEPTH=8 template instantiation:
//   pixel   -> uint8_t, pixel4 -> uint32_t
//   AV_RN4PA -> AV_RN32A, AV_WN4PA -> AV_WN32A
//   PIXEL_SPLAT_X4(x) -> ((x) * 0x01010101u)
//   CLIP -> av_clip_uint8
//   stride shift `_stride>>(sizeof(pixel)-1)` -> `_stride` (shift by 0 for 8-bit)
//   `N * sizeof(pixel)` -> `N`, `1<<(BIT_DEPTH-1)` -> 128
//   av_unused / av_log / av_assert dropped
//
// Only the reachable-for-SVQ3 subset is emitted: pred4x4[12], pred16x16[7],
// pred8x8[7] (chroma, 4:2:0 / chroma_format_idc<=1). The 8x8l predictors and the
// non-SVQ3 codec overrides (rv40/vp8) are omitted as unreachable.

// ─── mode enums (from h264pred.h) ──────────────────────────────────────────────
enum { VERT_PRED=0, HOR_PRED, DC_PRED, DIAG_DOWN_LEFT_PRED, DIAG_DOWN_RIGHT_PRED,
       VERT_RIGHT_PRED, HOR_DOWN_PRED, VERT_LEFT_PRED, HOR_UP_PRED,
       LEFT_DC_PRED, TOP_DC_PRED, DC_128_PRED };
enum { DC_PRED8x8=0, HOR_PRED8x8, VERT_PRED8x8, PLANE_PRED8x8,
       LEFT_DC_PRED8x8, TOP_DC_PRED8x8, DC_128_PRED8x8,
       ALZHEIMER_DC_L0T_PRED8x8, ALZHEIMER_DC_0LT_PRED8x8,
       ALZHEIMER_DC_L00_PRED8x8, ALZHEIMER_DC_0L0_PRED8x8 };

typedef void (*pred4x4_fn)(uint8_t* src, const uint8_t* topright, ptrdiff_t stride);
typedef void (*pred16_fn)(uint8_t* src, ptrdiff_t stride);

// ─── LOAD_*_EDGE macros (from h264pred_template.c) ─────────────────────────────
// av_unused kept: it is a pure-compiler attribute (not a framework helper) and
// matches the originals, whose edge loads are deliberately partially unread.
#define av_unused __attribute__((unused))
#define LOAD_TOP_RIGHT_EDGE\
    av_unused const unsigned t4 = topright[0];\
    av_unused const unsigned t5 = topright[1];\
    av_unused const unsigned t6 = topright[2];\
    av_unused const unsigned t7 = topright[3];\

#define LOAD_DOWN_LEFT_EDGE\
    av_unused const unsigned l4 = src[-1+4*stride];\
    av_unused const unsigned l5 = src[-1+5*stride];\
    av_unused const unsigned l6 = src[-1+6*stride];\
    av_unused const unsigned l7 = src[-1+7*stride];\

#define LOAD_LEFT_EDGE\
    av_unused const unsigned l0 = src[-1+0*stride];\
    av_unused const unsigned l1 = src[-1+1*stride];\
    av_unused const unsigned l2 = src[-1+2*stride];\
    av_unused const unsigned l3 = src[-1+3*stride];\

#define LOAD_TOP_EDGE\
    av_unused const unsigned t0 = src[ 0-1*stride];\
    av_unused const unsigned t1 = src[ 1-1*stride];\
    av_unused const unsigned t2 = src[ 2-1*stride];\
    av_unused const unsigned t3 = src[ 3-1*stride];\

// ─── 4x4 predictors ────────────────────────────────────────────────────────────
static void svq3_pred4x4_vertical_c(uint8_t *_src, const uint8_t *topright,
                                    ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const uint32_t a= AV_RN32A(src-stride);

    AV_WN32A(src+0*stride, a);
    AV_WN32A(src+1*stride, a);
    AV_WN32A(src+2*stride, a);
    AV_WN32A(src+3*stride, a);
}

static void svq3_pred4x4_horizontal_c(uint8_t *_src, const uint8_t *topright,
                                      ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    AV_WN32A(src+0*stride, (src[-1+0*stride]) * 0x01010101u);
    AV_WN32A(src+1*stride, (src[-1+1*stride]) * 0x01010101u);
    AV_WN32A(src+2*stride, (src[-1+2*stride]) * 0x01010101u);
    AV_WN32A(src+3*stride, (src[-1+3*stride]) * 0x01010101u);
}

static void svq3_pred4x4_dc_c(uint8_t *_src, const uint8_t *topright,
                              ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const int dc= (  src[-stride] + src[1-stride] + src[2-stride] + src[3-stride]
                   + src[-1+0*stride] + src[-1+1*stride] + src[-1+2*stride] + src[-1+3*stride] + 4) >>3;
    const uint32_t a = (dc) * 0x01010101u;

    AV_WN32A(src+0*stride, a);
    AV_WN32A(src+1*stride, a);
    AV_WN32A(src+2*stride, a);
    AV_WN32A(src+3*stride, a);
}

static void svq3_pred4x4_left_dc_c(uint8_t *_src, const uint8_t *topright,
                                   ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const int dc= (  src[-1+0*stride] + src[-1+1*stride] + src[-1+2*stride] + src[-1+3*stride] + 2) >>2;
    const uint32_t a = (dc) * 0x01010101u;

    AV_WN32A(src+0*stride, a);
    AV_WN32A(src+1*stride, a);
    AV_WN32A(src+2*stride, a);
    AV_WN32A(src+3*stride, a);
}

static void svq3_pred4x4_top_dc_c(uint8_t *_src, const uint8_t *topright,
                                  ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const int dc= (  src[-stride] + src[1-stride] + src[2-stride] + src[3-stride] + 2) >>2;
    const uint32_t a = (dc) * 0x01010101u;

    AV_WN32A(src+0*stride, a);
    AV_WN32A(src+1*stride, a);
    AV_WN32A(src+2*stride, a);
    AV_WN32A(src+3*stride, a);
}

static void svq3_pred4x4_128_dc_c(uint8_t *_src, const uint8_t *topright,
                                  ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const uint32_t a = (1<<7) * 0x01010101u;

    AV_WN32A(src+0*stride, a);
    AV_WN32A(src+1*stride, a);
    AV_WN32A(src+2*stride, a);
    AV_WN32A(src+3*stride, a);
}

static void svq3_pred4x4_down_right_c(uint8_t *_src, const uint8_t *topright,
                                      ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const int lt= src[-1-1*stride];
    LOAD_TOP_EDGE
    LOAD_LEFT_EDGE

    src[0+3*stride]=(l3 + 2*l2 + l1 + 2)>>2;
    src[0+2*stride]=
    src[1+3*stride]=(l2 + 2*l1 + l0 + 2)>>2;
    src[0+1*stride]=
    src[1+2*stride]=
    src[2+3*stride]=(l1 + 2*l0 + lt + 2)>>2;
    src[0+0*stride]=
    src[1+1*stride]=
    src[2+2*stride]=
    src[3+3*stride]=(l0 + 2*lt + t0 + 2)>>2;
    src[1+0*stride]=
    src[2+1*stride]=
    src[3+2*stride]=(lt + 2*t0 + t1 + 2)>>2;
    src[2+0*stride]=
    src[3+1*stride]=(t0 + 2*t1 + t2 + 2)>>2;
    src[3+0*stride]=(t1 + 2*t2 + t3 + 2)>>2;
}

// SVQ3 override of DIAG_DOWN_LEFT_PRED (h264pred.c: pred4x4_down_left_svq3_c)
static void svq3_pred4x4_down_left_svq3_c(uint8_t *src, const uint8_t *topright,
                                          ptrdiff_t stride)
{
    LOAD_TOP_EDGE
    LOAD_LEFT_EDGE

    src[0+0*stride]=(l1 + t1)>>1;
    src[1+0*stride]=
    src[0+1*stride]=(l2 + t2)>>1;
    src[2+0*stride]=
    src[1+1*stride]=
    src[0+2*stride]=
    src[3+0*stride]=
    src[2+1*stride]=
    src[1+2*stride]=
    src[0+3*stride]=
    src[3+1*stride]=
    src[2+2*stride]=
    src[1+3*stride]=
    src[3+2*stride]=
    src[2+3*stride]=
    src[3+3*stride]=(l3 + t3)>>1;
}

static void svq3_pred4x4_vertical_right_c(uint8_t *_src,
                                          const uint8_t *topright,
                                          ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const int lt= src[-1-1*stride];
    LOAD_TOP_EDGE
    LOAD_LEFT_EDGE

    src[0+0*stride]=
    src[1+2*stride]=(lt + t0 + 1)>>1;
    src[1+0*stride]=
    src[2+2*stride]=(t0 + t1 + 1)>>1;
    src[2+0*stride]=
    src[3+2*stride]=(t1 + t2 + 1)>>1;
    src[3+0*stride]=(t2 + t3 + 1)>>1;
    src[0+1*stride]=
    src[1+3*stride]=(l0 + 2*lt + t0 + 2)>>2;
    src[1+1*stride]=
    src[2+3*stride]=(lt + 2*t0 + t1 + 2)>>2;
    src[2+1*stride]=
    src[3+3*stride]=(t0 + 2*t1 + t2 + 2)>>2;
    src[3+1*stride]=(t1 + 2*t2 + t3 + 2)>>2;
    src[0+2*stride]=(lt + 2*l0 + l1 + 2)>>2;
    src[0+3*stride]=(l0 + 2*l1 + l2 + 2)>>2;
}

static void svq3_pred4x4_horizontal_down_c(uint8_t *_src,
                                           const uint8_t *topright,
                                           ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    const int lt= src[-1-1*stride];
    LOAD_TOP_EDGE
    LOAD_LEFT_EDGE

    src[0+0*stride]=
    src[2+1*stride]=(lt + l0 + 1)>>1;
    src[1+0*stride]=
    src[3+1*stride]=(l0 + 2*lt + t0 + 2)>>2;
    src[2+0*stride]=(lt + 2*t0 + t1 + 2)>>2;
    src[3+0*stride]=(t0 + 2*t1 + t2 + 2)>>2;
    src[0+1*stride]=
    src[2+2*stride]=(l0 + l1 + 1)>>1;
    src[1+1*stride]=
    src[3+2*stride]=(lt + 2*l0 + l1 + 2)>>2;
    src[0+2*stride]=
    src[2+3*stride]=(l1 + l2+ 1)>>1;
    src[1+2*stride]=
    src[3+3*stride]=(l0 + 2*l1 + l2 + 2)>>2;
    src[0+3*stride]=(l2 + l3 + 1)>>1;
    src[1+3*stride]=(l1 + 2*l2 + l3 + 2)>>2;
}

static void svq3_pred4x4_vertical_left_c(uint8_t *_src,
                                         const uint8_t *_topright,
                                         ptrdiff_t _stride)
{
    uint8_t *src = _src;
    const uint8_t *topright = _topright;
    int stride = _stride;
    LOAD_TOP_EDGE
    LOAD_TOP_RIGHT_EDGE

    src[0+0*stride]=(t0 + t1 + 1)>>1;
    src[1+0*stride]=
    src[0+2*stride]=(t1 + t2 + 1)>>1;
    src[2+0*stride]=
    src[1+2*stride]=(t2 + t3 + 1)>>1;
    src[3+0*stride]=
    src[2+2*stride]=(t3 + t4+ 1)>>1;
    src[3+2*stride]=(t4 + t5+ 1)>>1;
    src[0+1*stride]=(t0 + 2*t1 + t2 + 2)>>2;
    src[1+1*stride]=
    src[0+3*stride]=(t1 + 2*t2 + t3 + 2)>>2;
    src[2+1*stride]=
    src[1+3*stride]=(t2 + 2*t3 + t4 + 2)>>2;
    src[3+1*stride]=
    src[2+3*stride]=(t3 + 2*t4 + t5 + 2)>>2;
    src[3+3*stride]=(t4 + 2*t5 + t6 + 2)>>2;
}

static void svq3_pred4x4_horizontal_up_c(uint8_t *_src, const uint8_t *topright,
                                         ptrdiff_t _stride)
{
    uint8_t *src = _src;
    int stride = _stride;
    LOAD_LEFT_EDGE

    src[0+0*stride]=(l0 + l1 + 1)>>1;
    src[1+0*stride]=(l0 + 2*l1 + l2 + 2)>>2;
    src[2+0*stride]=
    src[0+1*stride]=(l1 + l2 + 1)>>1;
    src[3+0*stride]=
    src[1+1*stride]=(l1 + 2*l2 + l3 + 2)>>2;
    src[2+1*stride]=
    src[0+2*stride]=(l2 + l3 + 1)>>1;
    src[3+1*stride]=
    src[1+2*stride]=(l2 + 2*l3 + l3 + 2)>>2;
    src[3+2*stride]=
    src[1+3*stride]=
    src[0+3*stride]=
    src[2+2*stride]=
    src[2+3*stride]=
    src[3+3*stride]=l3;
}

// ─── 16x16 predictors ──────────────────────────────────────────────────────────
static void svq3_pred16x16_vertical_c(uint8_t *_src, ptrdiff_t _stride)
{
    int i;
    uint8_t *src = _src;
    int stride = _stride;
    const uint32_t a = AV_RN32A(((uint32_t*)(src-stride))+0);
    const uint32_t b = AV_RN32A(((uint32_t*)(src-stride))+1);
    const uint32_t c = AV_RN32A(((uint32_t*)(src-stride))+2);
    const uint32_t d = AV_RN32A(((uint32_t*)(src-stride))+3);

    for(i=0; i<16; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, b);
        AV_WN32A(((uint32_t*)(src+i*stride))+2, c);
        AV_WN32A(((uint32_t*)(src+i*stride))+3, d);
    }
}

static void svq3_pred16x16_horizontal_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    uint8_t *src = _src;

    for(i=0; i<16; i++){
        const uint32_t a = (src[-1+i*stride]) * 0x01010101u;

        AV_WN32A(((uint32_t*)(src+i*stride))+0, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+2, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+3, a);
    }
}

#define PREDICT_16x16_DC(v)\
    for(i=0; i<16; i++){\
        AV_WN32A(src+ 0, v);\
        AV_WN32A(src+ 4, v);\
        AV_WN32A(src+ 8, v);\
        AV_WN32A(src+12, v);\
        src += stride;\
    }

static void svq3_pred16x16_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i, dc=0;
    uint8_t *src = _src;
    uint32_t dcsplat;

    for(i=0;i<16; i++){
        dc+= src[-1+i*stride];
    }

    for(i=0;i<16; i++){
        dc+= src[i-stride];
    }

    dcsplat = ((dc+16)>>5) * 0x01010101u;
    PREDICT_16x16_DC(dcsplat);
}

static void svq3_pred16x16_left_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i, dc=0;
    uint8_t *src = _src;
    uint32_t dcsplat;

    for(i=0;i<16; i++){
        dc+= src[-1+i*stride];
    }

    dcsplat = ((dc+8)>>4) * 0x01010101u;
    PREDICT_16x16_DC(dcsplat);
}

static void svq3_pred16x16_top_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i, dc=0;
    uint8_t *src = _src;
    uint32_t dcsplat;

    for(i=0;i<16; i++){
        dc+= src[i-stride];
    }

    dcsplat = ((dc+8)>>4) * 0x01010101u;
    PREDICT_16x16_DC(dcsplat);
}

static void svq3_pred16x16_128_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    uint8_t *src = _src;
    PREDICT_16x16_DC(((1<<7)+0) * 0x01010101u);
}

// SVQ3 override of PLANE_PRED8x8 (h264pred.c: pred16x16_plane_svq3_c ==
//   pred16x16_plane_compat_8_c(src, stride, /*svq3=*/1, /*rv40=*/0))
static void svq3_pred16x16_plane_svq3_c(uint8_t *_src, ptrdiff_t _stride)
{
  const int svq3 = 1;
  const int rv40 = 0;
  int i, j, k;
  int a;
  uint8_t *src = _src;
  int stride = _stride;
  const uint8_t * const src0 = src +7-stride;
  const uint8_t *       src1 = src +8*stride-1;
  const uint8_t *       src2 = src1-2*stride;    // == src+6*stride-1;
  int H = src0[1] - src0[-1];
  int V = src1[0] - src2[ 0];
  for(k=2; k<=8; ++k) {
    src1 += stride; src2 -= stride;
    H += k*(src0[k] - src0[-k]);
    V += k*(src1[0] - src2[ 0]);
  }
  if(svq3){
    H = ( 5*(H/4) ) / 16;
    V = ( 5*(V/4) ) / 16;

    /* required for 100% accuracy */
    i = H; H = V; V = i;
  }else if(rv40){
    H = ( H + (H>>2) ) >> 4;
    V = ( V + (V>>2) ) >> 4;
  }else{
    H = ( 5*H+32 ) >> 6;
    V = ( 5*V+32 ) >> 6;
  }

  a = 16*(src1[0] + src2[16] + 1) - 7*(V+H);
  for(j=16; j>0; --j) {
    int b = a;
    a += V;
    for(i=-16; i<0; i+=4) {
      src[16+i] = av_clip_uint8((b    ) >> 5);
      src[17+i] = av_clip_uint8((b+  H) >> 5);
      src[18+i] = av_clip_uint8((b+2*H) >> 5);
      src[19+i] = av_clip_uint8((b+3*H) >> 5);
      b += 4*H;
    }
    src += stride;
  }
}

// ─── 8x8 chroma predictors (chroma_format_idc <= 1 / 4:2:0) ────────────────────
static void svq3_pred8x8_vertical_c(uint8_t *_src, ptrdiff_t _stride)
{
    int i;
    uint8_t *src = _src;
    int stride = _stride;
    const uint32_t a= AV_RN32A(((uint32_t*)(src-stride))+0);
    const uint32_t b= AV_RN32A(((uint32_t*)(src-stride))+1);

    for(i=0; i<8; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, b);
    }
}

static void svq3_pred8x8_horizontal_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    uint8_t *src = _src;

    for(i=0; i<8; i++){
        const uint32_t a = (src[-1+i*stride]) * 0x01010101u;
        AV_WN32A(((uint32_t*)(src+i*stride))+0, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, a);
    }
}

static void svq3_pred8x8_128_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    const uint32_t a = ((1<<7)+0) * 0x01010101u;
    uint8_t *src = _src;
    for(i=0; i<8; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, a);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, a);
    }
}

static void svq3_pred8x8_left_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    int dc0, dc2;
    uint32_t dc0splat, dc2splat;
    uint8_t *src = _src;

    dc0=dc2=0;
    for(i=0;i<4; i++){
        dc0+= src[-1+i*stride];
        dc2+= src[-1+(i+4)*stride];
    }
    dc0splat = ((dc0 + 2)>>2) * 0x01010101u;
    dc2splat = ((dc2 + 2)>>2) * 0x01010101u;

    for(i=0; i<4; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, dc0splat);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, dc0splat);
    }
    for(i=4; i<8; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, dc2splat);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, dc2splat);
    }
}

static void svq3_pred8x8_top_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    int dc0, dc1;
    uint32_t dc0splat, dc1splat;
    uint8_t *src = _src;

    dc0=dc1=0;
    for(i=0;i<4; i++){
        dc0+= src[i-stride];
        dc1+= src[4+i-stride];
    }
    dc0splat = ((dc0 + 2)>>2) * 0x01010101u;
    dc1splat = ((dc1 + 2)>>2) * 0x01010101u;

    for(i=0; i<4; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, dc0splat);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, dc1splat);
    }
    for(i=4; i<8; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, dc0splat);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, dc1splat);
    }
}

static void svq3_pred8x8_dc_c(uint8_t *_src, ptrdiff_t stride)
{
    int i;
    int dc0, dc1, dc2;
    uint32_t dc0splat, dc1splat, dc2splat, dc3splat;
    uint8_t *src = _src;

    dc0=dc1=dc2=0;
    for(i=0;i<4; i++){
        dc0+= src[-1+i*stride] + src[i-stride];
        dc1+= src[4+i-stride];
        dc2+= src[-1+(i+4)*stride];
    }
    dc0splat = ((dc0 + 4)>>3) * 0x01010101u;
    dc1splat = ((dc1 + 2)>>2) * 0x01010101u;
    dc2splat = ((dc2 + 2)>>2) * 0x01010101u;
    dc3splat = ((dc1 + dc2 + 4)>>3) * 0x01010101u;

    for(i=0; i<4; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, dc0splat);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, dc1splat);
    }
    for(i=4; i<8; i++){
        AV_WN32A(((uint32_t*)(src+i*stride))+0, dc2splat);
        AV_WN32A(((uint32_t*)(src+i*stride))+1, dc3splat);
    }
}

static void svq3_pred8x8_plane_c(uint8_t *_src, ptrdiff_t _stride)
{
  int j, k;
  int a;
  uint8_t *src = _src;
  int stride = _stride;
  const uint8_t * const src0 = src +3-stride;
  const uint8_t *       src1 = src +4*stride-1;
  const uint8_t *       src2 = src1-2*stride;    // == src+2*stride-1;
  int H = src0[1] - src0[-1];
  int V = src1[0] - src2[ 0];
  for(k=2; k<=4; ++k) {
    src1 += stride; src2 -= stride;
    H += k*(src0[k] - src0[-k]);
    V += k*(src1[0] - src2[ 0]);
  }
  H = ( 17*H+16 ) >> 5;
  V = ( 17*V+16 ) >> 5;

  a = 16*(src1[0] + src2[8]+1) - 3*(V+H);
  for(j=8; j>0; --j) {
    int b = a;
    a += V;
    src[0] = av_clip_uint8((b    ) >> 5);
    src[1] = av_clip_uint8((b+  H) >> 5);
    src[2] = av_clip_uint8((b+2*H) >> 5);
    src[3] = av_clip_uint8((b+3*H) >> 5);
    src[4] = av_clip_uint8((b+4*H) >> 5);
    src[5] = av_clip_uint8((b+5*H) >> 5);
    src[6] = av_clip_uint8((b+6*H) >> 5);
    src[7] = av_clip_uint8((b+7*H) >> 5);
    src += stride;
  }
}

#undef PREDICT_16x16_DC
#undef LOAD_TOP_RIGHT_EDGE
#undef LOAD_DOWN_LEFT_EDGE
#undef LOAD_LEFT_EDGE
#undef LOAD_TOP_EDGE
#undef av_unused

// ─── dispatch tables (reachable part of ff_h264_pred_init for SVQ3, 8-bit) ─────
static pred4x4_fn svq3_pred4x4[12];
static pred16_fn  svq3_pred16x16[7];
static pred16_fn  svq3_pred8x8[7];   // chroma; modes 7-10 unreachable -> NULL

static void svq3_pred_init(void)
{
    // pred4x4 (H264_PRED(8), then SVQ3 DIAG_DOWN_LEFT override)
    svq3_pred4x4[VERT_PRED           ] = svq3_pred4x4_vertical_c;
    svq3_pred4x4[HOR_PRED            ] = svq3_pred4x4_horizontal_c;
    svq3_pred4x4[DC_PRED             ] = svq3_pred4x4_dc_c;
    svq3_pred4x4[DIAG_DOWN_LEFT_PRED ] = svq3_pred4x4_down_left_svq3_c; // SVQ3 override
    svq3_pred4x4[DIAG_DOWN_RIGHT_PRED] = svq3_pred4x4_down_right_c;
    svq3_pred4x4[VERT_RIGHT_PRED     ] = svq3_pred4x4_vertical_right_c;
    svq3_pred4x4[HOR_DOWN_PRED       ] = svq3_pred4x4_horizontal_down_c;
    svq3_pred4x4[VERT_LEFT_PRED      ] = svq3_pred4x4_vertical_left_c;
    svq3_pred4x4[HOR_UP_PRED         ] = svq3_pred4x4_horizontal_up_c;
    svq3_pred4x4[LEFT_DC_PRED        ] = svq3_pred4x4_left_dc_c;
    svq3_pred4x4[TOP_DC_PRED         ] = svq3_pred4x4_top_dc_c;
    svq3_pred4x4[DC_128_PRED         ] = svq3_pred4x4_128_dc_c; // set: codec_id != VP8

    // pred8x8 (chroma_format_idc <= 1, i.e. 4:2:0)
    svq3_pred8x8[DC_PRED8x8     ] = svq3_pred8x8_dc_c;
    svq3_pred8x8[HOR_PRED8x8    ] = svq3_pred8x8_horizontal_c;
    svq3_pred8x8[VERT_PRED8x8   ] = svq3_pred8x8_vertical_c;
    svq3_pred8x8[PLANE_PRED8x8  ] = svq3_pred8x8_plane_c;
    svq3_pred8x8[LEFT_DC_PRED8x8] = svq3_pred8x8_left_dc_c;
    svq3_pred8x8[TOP_DC_PRED8x8 ] = svq3_pred8x8_top_dc_c;
    svq3_pred8x8[DC_128_PRED8x8 ] = svq3_pred8x8_128_dc_c;

    // pred16x16 (then SVQ3 PLANE override)
    svq3_pred16x16[DC_PRED8x8     ] = svq3_pred16x16_dc_c;
    svq3_pred16x16[VERT_PRED8x8   ] = svq3_pred16x16_vertical_c;
    svq3_pred16x16[HOR_PRED8x8    ] = svq3_pred16x16_horizontal_c;
    svq3_pred16x16[PLANE_PRED8x8  ] = svq3_pred16x16_plane_svq3_c; // SVQ3 override
    svq3_pred16x16[LEFT_DC_PRED8x8] = svq3_pred16x16_left_dc_c;
    svq3_pred16x16[TOP_DC_PRED8x8 ] = svq3_pred16x16_top_dc_c;
    svq3_pred16x16[DC_128_PRED8x8 ] = svq3_pred16x16_128_dc_c;
}

// ─── check_intra helpers (h264_parse.c; logctx dropped, args reordered per contract) ──
static int ff_h264_check_intra_pred_mode(int top_samples_available,
                                         int left_samples_available,
                                         int mode, int is_chroma)
{
    static const int8_t top[4]  = { LEFT_DC_PRED8x8, 1, -1, -1 };
    static const int8_t left[5] = { TOP_DC_PRED8x8, -1,  2, -1, DC_128_PRED8x8 };

    if (mode > 3U) {
        // av_log: out of range intra chroma pred mode
        return -1; // AVERROR_INVALIDDATA
    }

    if (!(top_samples_available & 0x8000)) {
        mode = top[mode];
        if (mode < 0) {
            // av_log: top block unavailable for requested intra mode
            return -1; // AVERROR_INVALIDDATA
        }
    }

    if ((left_samples_available & 0x8080) != 0x8080) {
        mode = left[mode];
        if (mode < 0) {
            // av_log: left block unavailable for requested intra mode
            return -1; // AVERROR_INVALIDDATA
        }
        if (is_chroma && (left_samples_available & 0x8080)) {
            // mad cow disease mode, aka MBAFF + constrained_intra_pred
            mode = ALZHEIMER_DC_L0T_PRED8x8 +
                   (!(left_samples_available & 0x8000)) +
                   2 * (mode == DC_128_PRED8x8);
        }
    }

    return mode;
}

static int ff_h264_check_intra4x4_pred_mode(int8_t *pred_mode_cache,
                                            int top_samples_available,
                                            int left_samples_available)
{
    static const int8_t top[12] = {
        -1, 0, LEFT_DC_PRED, -1, -1, -1, -1, -1, 0
    };
    static const int8_t left[12] = {
        0, -1, TOP_DC_PRED, 0, -1, -1, -1, 0, -1, DC_128_PRED
    };
    int i;

    if (!(top_samples_available & 0x8000)) {
        for (i = 0; i < 4; i++) {
            int status = top[pred_mode_cache[scan8[0] + i]];
            if (status < 0) {
                // av_log: top block unavailable for requested intra mode
                return -1; // AVERROR_INVALIDDATA
            } else if (status) {
                pred_mode_cache[scan8[0] + i] = status;
            }
        }
    }

    if ((left_samples_available & 0x8888) != 0x8888) {
        static const int mask[4] = { 0x8000, 0x2000, 0x80, 0x20 };
        for (i = 0; i < 4; i++)
            if (!(left_samples_available & mask[i])) {
                int status = left[pred_mode_cache[scan8[0] + 8 * i]];
                if (status < 0) {
                    // av_log: left block unavailable for requested intra4x4 mode
                    return -1; // AVERROR_INVALIDDATA
                } else if (status) {
                    pred_mode_cache[scan8[0] + 8 * i] = status;
                }
            }
    }

    return 0;
}
