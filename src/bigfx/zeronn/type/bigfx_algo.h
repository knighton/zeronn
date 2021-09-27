#ifndef ZERONN_TYPE_BIGFX_ALGO_H_
#define ZERONN_TYPE_BIGFX_ALGO_H_

#include <stdint.h>

#include "zeronn/type/bigfx_base.h"
#include "zeronn/type/bigfx_state.h"

bigfx_t bigfx_add(bigfx_t a, bigfx_t b);
bigfx_t bigfx_incr(bigfx_t a);

bigfx_t bigfx_sub(bigfx_t a, bigfx_t b);
bigfx_t bigfx_decr(bigfx_t a);

bigfx_t bigfx_mul(bigfx_t a, bigfx_t b);
bigfx_t bigfx_mul_log2(bigfx_t a, uint8_t b);

bigfx_t bigfx_div(bigfx_t a, bigfx_t b);
bigfx_t bigfx_div_log2(bigfx_t a, uint8_t b);

bigfx_t bigint_div(bigfx_t a, bigfx_t b);

bigfx_t bigfx_log2(bigfx_t a);
bigfx_t bigfx_log10(bigfx_t a);

#endif /* ZERONN_TYPE_BIGFX_ALGO_H_ */
