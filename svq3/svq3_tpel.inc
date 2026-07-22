// Third-pel motion compensation. Ported VERBATIM from ffmpeg release/8.1: libavcodec/tpeldsp.c (kernels) + libavcodec/pel_template.c (mc00 copy deps, defined by the hpel group).
//
// The nine put_/avg_tpel_pixels_mcXY_c kernels and the two [11]-entry dispatch
// tables are transliterated exactly. The framework (TpelDSPContext, av_cold,
// arch init dispatch) is stripped; the table-fill body of ff_tpeldsp_init lives
// in svq3_tpel_init().
//
// The mc00 kernels delegate to the byte-copy helpers put_pixels{2,4,8,16}_8_c /
// avg_pixels{2,4,8,16}_8_c. In ffmpeg those come from pel_template.c's DEF_PEL,
// which is #included by BOTH hpeldsp.c and tpeldsp.c. Here that single
// definition is emitted by the HALF-PEL group (svq3_hpel.inc). We only
// forward-declare them below so this fragment compiles independent of include
// order; the one definition must come from the hpel group in the same TU.

// --- byte-copy helpers owned by the hpel group (pel_template.c / DEF_PEL) ---

typedef void (*tpel_fn)(uint8_t* dst, const uint8_t* src, int stride, int width, int height);

// ---- put ----

static inline void put_tpel_pixels_mc00_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    switch (width) {
    case 2:
        svq3_put_pixels2_8_c(dst, src, stride, height);
        break;
    case 4:
        svq3_put_pixels4_8_c(dst, src, stride, height);
        break;
    case 8:
        svq3_put_pixels8_8_c(dst, src, stride, height);
        break;
    case 16:
        svq3_put_pixels16_8_c(dst, src, stride, height);
        break;
    }
}

static inline void put_tpel_pixels_mc10_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((2 * src[j] + src[j + 1] + 1) *
                      683) >> 11;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc20_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((src[j] + 2 * src[j + 1] + 1) *
                      683) >> 11;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc01_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((2 * src[j] + src[j + stride] + 1) *
                      683) >> 11;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc11_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((4 * src[j]          + 3 * src[j + 1] +
                       3 * src[j + stride] + 2 * src[j + stride + 1] + 6) *
                      2731) >> 15;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc12_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((3 * src[j]          + 2 * src[j + 1] +
                       4 * src[j + stride] + 3 * src[j + stride + 1] + 6) *
                      2731) >> 15;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc02_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((src[j] + 2 * src[j + stride] + 1) *
                      683) >> 11;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc21_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((3 * src[j]          + 4 * src[j + 1] +
                       2 * src[j + stride] + 3 * src[j + stride + 1] + 6) *
                      2731) >> 15;
        src += stride;
        dst += stride;
    }
}

static inline void put_tpel_pixels_mc22_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = ((2 * src[j]          + 3 * src[j + 1] +
                       3 * src[j + stride] + 4 * src[j + stride + 1] + 6) *
                      2731) >> 15;
        src += stride;
        dst += stride;
    }
}

// ---- avg ----

static inline void avg_tpel_pixels_mc00_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    switch (width) {
    case 2:
        svq3_avg_pixels2_8_c(dst, src, stride, height);
        break;
    case 4:
        svq3_avg_pixels4_8_c(dst, src, stride, height);
        break;
    case 8:
        svq3_avg_pixels8_8_c(dst, src, stride, height);
        break;
    case 16:
        svq3_avg_pixels16_8_c(dst, src, stride, height);
        break;
    }
}

static inline void avg_tpel_pixels_mc10_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((2 * src[j] + src[j + 1] + 1) *
                        683) >> 11) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc20_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((src[j] + 2 * src[j + 1] + 1) *
                        683) >> 11) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc01_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((2 * src[j] + src[j + stride] + 1) *
                        683) >> 11) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc11_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((4 * src[j]          + 3 * src[j + 1] +
                         3 * src[j + stride] + 2 * src[j + stride + 1] + 6) *
                        2731) >> 15) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc12_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((3 * src[j]          + 2 * src[j + 1] +
                         4 * src[j + stride] + 3 * src[j + stride + 1] + 6) *
                        2731) >> 15) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc02_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((src[j] + 2 * src[j + stride] + 1) *
                        683) >> 11) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc21_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((3 * src[j]          + 4 * src[j + 1] +
                         2 * src[j + stride] + 3 * src[j + stride + 1] + 6) *
                        2731) >> 15) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

static inline void avg_tpel_pixels_mc22_c(uint8_t *dst, const uint8_t *src,
                                          int stride, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++)
            dst[j] = (dst[j] +
                      (((2 * src[j]          + 3 * src[j + 1] +
                         3 * src[j + stride] + 4 * src[j + stride + 1] + 6) *
                        2731) >> 15) + 1) >> 1;
        src += stride;
        dst += stride;
    }
}

// ---- dispatch tables (dxy = xthird + 4*ythird); indices 3 and 7 stay NULL ----

static tpel_fn svq3_put_tpel_pixels_tab[11];
static tpel_fn svq3_avg_tpel_pixels_tab[11];

static void svq3_tpel_init(void)
{
    svq3_put_tpel_pixels_tab[ 0] = put_tpel_pixels_mc00_c;
    svq3_put_tpel_pixels_tab[ 1] = put_tpel_pixels_mc10_c;
    svq3_put_tpel_pixels_tab[ 2] = put_tpel_pixels_mc20_c;
    svq3_put_tpel_pixels_tab[ 4] = put_tpel_pixels_mc01_c;
    svq3_put_tpel_pixels_tab[ 5] = put_tpel_pixels_mc11_c;
    svq3_put_tpel_pixels_tab[ 6] = put_tpel_pixels_mc21_c;
    svq3_put_tpel_pixels_tab[ 8] = put_tpel_pixels_mc02_c;
    svq3_put_tpel_pixels_tab[ 9] = put_tpel_pixels_mc12_c;
    svq3_put_tpel_pixels_tab[10] = put_tpel_pixels_mc22_c;

    svq3_avg_tpel_pixels_tab[ 0] = avg_tpel_pixels_mc00_c;
    svq3_avg_tpel_pixels_tab[ 1] = avg_tpel_pixels_mc10_c;
    svq3_avg_tpel_pixels_tab[ 2] = avg_tpel_pixels_mc20_c;
    svq3_avg_tpel_pixels_tab[ 4] = avg_tpel_pixels_mc01_c;
    svq3_avg_tpel_pixels_tab[ 5] = avg_tpel_pixels_mc11_c;
    svq3_avg_tpel_pixels_tab[ 6] = avg_tpel_pixels_mc21_c;
    svq3_avg_tpel_pixels_tab[ 8] = avg_tpel_pixels_mc02_c;
    svq3_avg_tpel_pixels_tab[ 9] = avg_tpel_pixels_mc12_c;
    svq3_avg_tpel_pixels_tab[10] = avg_tpel_pixels_mc22_c;
}
