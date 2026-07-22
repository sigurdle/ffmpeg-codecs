/*
    Independent reference: FFmpeg's ff_simple_idct8 (8-bit) copied VERBATIM from
    libavcodec/simple_idct_template.c (BIT_DEPTH==8, IN_IDCT_DEPTH==16, non-64bit DC
    path), so it cannot share a transcription error with our optimized simple_idct8.h.

    Reached only through the _mp4v_ref_idct test hook (M0c cross-checks our IDCT
    against this on random + adversarial blocks); the decode path never calls it. Like
    the other _mp4v_test_* hooks (and mpeg/vp6's _*_test_idct), it is exported in both
    builds, so it rides along in the shipped binary — a few KB, one idct8 function.
    Keeping ffmpeg's exact C IDCT here is the canonical conformance baseline: the CLI's
    default build
    selects a SIMD IDCT (SSE2/AVX) that is allowed to differ by 1 LSB from this C
    reference on rare high-frequency blocks — to bit-compare against the real reference
    the CLI must be run with `-cpuflags 0`. LGPL (Niedermayer 2001).
*/
#ifndef MPEG4_IDCT_REF_H
#define MPEG4_IDCT_REF_H
#include <stdint.h>
#include <string.h>
namespace mp4ref {

#define W1 22725
#define W2 21407
#define W3 19266
#define W4 16383
#define W5 12873
#define W6 8867
#define W7 4520
#define ROW_SHIFT 11
#define COL_SHIFT 20
#define DC_SHIFT 3
#define SUINT unsigned
#define MUL(a,b)    ((a)*(b))
#define MAC(a,b,c)  ((a) += (b)*(c))

static inline uint32_t RN32(const int16_t* p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t RN64(const int16_t* p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline void WN32(int16_t* p, uint32_t v){ memcpy(p,&v,4); }
static inline int clip_u8(int v){ return v<0?0:(v>255?255:v); }

static inline void idctRowCondDC(int16_t* row){
    SUINT a0,a1,a2,a3,b0,b1,b2,b3;
    if(!(RN32(row+2) | RN32(row+4) | RN32(row+6) | row[1])){
        uint32_t temp = (row[0] * (1 << DC_SHIFT)) & 0xffff;
        temp += temp * (1 << 16);
        WN32(row, temp); WN32(row+2, temp); WN32(row+4, temp); WN32(row+6, temp);
        return;
    }
    a0 = ((SUINT)W4 * row[0]) + (1 << (ROW_SHIFT - 1));
    a1 = a0; a2 = a0; a3 = a0;
    a0 += (SUINT)W2 * row[2];
    a1 += (SUINT)W6 * row[2];
    a2 -= (SUINT)W6 * row[2];
    a3 -= (SUINT)W2 * row[2];
    b0 = MUL(W1, row[1]); MAC(b0, W3, row[3]);
    b1 = MUL(W3, row[1]); MAC(b1, -W7, row[3]);
    b2 = MUL(W5, row[1]); MAC(b2, -W1, row[3]);
    b3 = MUL(W7, row[1]); MAC(b3, -W5, row[3]);
    if(RN64(row+4)){
        a0 += (SUINT)  W4*row[4] + (SUINT)W6*row[6];
        a1 += (SUINT)- W4*row[4] - (SUINT)W2*row[6];
        a2 += (SUINT)- W4*row[4] + (SUINT)W2*row[6];
        a3 += (SUINT)  W4*row[4] - (SUINT)W6*row[6];
        MAC(b0,  W5, row[5]); MAC(b0,  W7, row[7]);
        MAC(b1, -W1, row[5]); MAC(b1, -W5, row[7]);
        MAC(b2,  W7, row[5]); MAC(b2,  W3, row[7]);
        MAC(b3,  W3, row[5]); MAC(b3, -W1, row[7]);
    }
    row[0] = (int)(a0 + b0) >> ROW_SHIFT;
    row[7] = (int)(a0 - b0) >> ROW_SHIFT;
    row[1] = (int)(a1 + b1) >> ROW_SHIFT;
    row[6] = (int)(a1 - b1) >> ROW_SHIFT;
    row[2] = (int)(a2 + b2) >> ROW_SHIFT;
    row[5] = (int)(a2 - b2) >> ROW_SHIFT;
    row[3] = (int)(a3 + b3) >> ROW_SHIFT;
    row[4] = (int)(a3 - b3) >> ROW_SHIFT;
}

#define IDCT_COLS do {                                  \
        a0 = (SUINT)W4 * (col[8*0] + ((1<<(COL_SHIFT-1))/W4)); \
        a1 = a0; a2 = a0; a3 = a0;                       \
        a0 += (SUINT) W2*col[8*2];                       \
        a1 += (SUINT) W6*col[8*2];                       \
        a2 += (SUINT)-W6*col[8*2];                       \
        a3 += (SUINT)-W2*col[8*2];                       \
        b0 = MUL(W1, col[8*1]);                          \
        b1 = MUL(W3, col[8*1]);                          \
        b2 = MUL(W5, col[8*1]);                          \
        b3 = MUL(W7, col[8*1]);                          \
        MAC(b0,  W3, col[8*3]);                          \
        MAC(b1, -W7, col[8*3]);                          \
        MAC(b2, -W1, col[8*3]);                          \
        MAC(b3, -W5, col[8*3]);                          \
        if (col[8*4]) {                                 \
            a0 += (SUINT) W4*col[8*4];                   \
            a1 += (SUINT)-W4*col[8*4];                   \
            a2 += (SUINT)-W4*col[8*4];                   \
            a3 += (SUINT) W4*col[8*4];                   \
        }                                               \
        if (col[8*5]) {                                 \
            MAC(b0,  W5, col[8*5]);                      \
            MAC(b1, -W1, col[8*5]);                      \
            MAC(b2,  W7, col[8*5]);                      \
            MAC(b3,  W3, col[8*5]);                      \
        }                                               \
        if (col[8*6]) {                                 \
            a0 += (SUINT) W6*col[8*6];                   \
            a1 += (SUINT)-W2*col[8*6];                   \
            a2 += (SUINT) W2*col[8*6];                   \
            a3 += (SUINT)-W6*col[8*6];                   \
        }                                               \
        if (col[8*7]) {                                 \
            MAC(b0,  W7, col[8*7]);                      \
            MAC(b1, -W5, col[8*7]);                      \
            MAC(b2,  W3, col[8*7]);                      \
            MAC(b3, -W1, col[8*7]);                      \
        }                                               \
    } while (0)

static inline void idctSparseCol(int16_t* col){
    SUINT a0,a1,a2,a3,b0,b1,b2,b3;
    IDCT_COLS;
    col[0 ] = (int)(a0 + b0) >> COL_SHIFT;
    col[8 ] = (int)(a1 + b1) >> COL_SHIFT;
    col[16] = (int)(a2 + b2) >> COL_SHIFT;
    col[24] = (int)(a3 + b3) >> COL_SHIFT;
    col[32] = (int)(a3 - b3) >> COL_SHIFT;
    col[40] = (int)(a2 - b2) >> COL_SHIFT;
    col[48] = (int)(a1 - b1) >> COL_SHIFT;
    col[56] = (int)(a0 - b0) >> COL_SHIFT;
}

static inline void idct8(int16_t* block){
    for(int i=0;i<8;i++) idctRowCondDC(block + i*8);
    for(int i=0;i<8;i++) idctSparseCol(block + i);
}

#undef W1
#undef W2
#undef W3
#undef W4
#undef W5
#undef W6
#undef W7
#undef ROW_SHIFT
#undef COL_SHIFT
#undef DC_SHIFT
#undef SUINT
#undef MUL
#undef MAC
#undef IDCT_COLS

} // namespace mp4ref
#endif
