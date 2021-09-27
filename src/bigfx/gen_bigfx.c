#include <math.h>
#include <stdio.h>

#include "zeronn/type/bigfx.h"

void gen_exp(float a, float* b, float* b2b) {
    *b = exp(a);
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_exp(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_exp_neg(float a, float* b, float* b2b) {
    gen_exp(-a, b, b2b);
}

void gen_log(float a, float* b, float* b2b) {
    *b = log(a);
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_log(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_log2(float a, float* b, float* b2b) {
    *b = log2(a);
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_log2(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_log10(float a, float* b, float* b2b) {
    *b = log10(a);
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_log10(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_recip(float a, float* b, float* b2b) {
    *b = 1 / a;
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_recip(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_sq(float a, float* b, float* b2b) {
    *b = a * a;
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_sq(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_sqrt(float a, float* b, float* b2b) {
    *b = sqrt(a);
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_sqrt(a2);
    *b2b = float_from_bigfx(b2);
}

void gen_tanh(float a, float* b, float* b2b) {
    *b = tanh(a);
    bigfx_t a2 = bigfx_from_float(a);
    bigfx_t b2 = bigfx_tanh(a2);
    *b2b = float_from_bigfx(b2);
}

typedef void (*gen_t)(float a, float* b, float* b2b);

static gen_t GENS[] = {
    gen_exp,
    gen_exp_neg,
    gen_log,
    gen_log2,
    gen_log10,
    gen_recip,
    gen_sq,
    gen_sqrt,
    gen_tanh,
};

#define NUM_GENS (sizeof(GENS) / sizeof(*GENS))

static char* NAMES[] = {
    "exp_pos",
    "exp_neg",
    "log",
    "log2",
    "log10",
    "recip",
    "sq",
    "sqrt",
    "tanh",
};

#define LO 1
#define HI (10 << FX_BITS)

int main(void) {
    for (int i = 0; i < (int)NUM_GENS; ++i) {
        const char* name = NAMES[i];
        gen_t gen = GENS[i];
        for (int j = LO; j < HI; ++j) {
            float a = (float)j / FX_ONE;
            float b;
            float b2b;
            gen(a, &b, &b2b);
            printf("%s %.6f %.6f %.6f\n", name, a, b, b2b);
        }
    }
}
