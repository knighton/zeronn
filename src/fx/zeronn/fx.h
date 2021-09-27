#ifndef ZERONN_FIXED_H_
#define ZERONN_FIXED_H_

#include <stdint.h>

#define FX_ONE (1 << 10)
#define FX_ONE_SQRT (1 << 5)

typedef int32_t fx_t;

float fp_from_fx(fx_t x);

fx_t fx_from_fp(float x);

fx_t fx_div(fx_t a, fx_t b);

int32_t i_sqrt(int32_t x);

fx_t fx_sqrt(fx_t x);

fx_t fx_sq(fx_t x);

fx_t fx_exp(fx_t x);

#endif  /* ZERONN_FIXED_H_ */
