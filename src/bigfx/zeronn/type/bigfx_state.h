#ifndef ZERONN_TYPE_BIGFX_STATE_H_
#define ZERONN_TYPE_BIGFX_STATE_H_

#include <stddef.h>

#include "zeronn/type/bigfx.h"

typedef struct {
    bool has_init;
    bigfx_t* table;
    bigfx_t recip_log2;
    bigfx_t recip_log10;
} bigfx_state_t;

extern bigfx_state_t BFX_STATE;

void bigfx_state_init(void);

bigfx_t bigfx_exp(bigfx_t a);
bigfx_t bigfx_log(bigfx_t a);
bigfx_t bigfx_recip(bigfx_t a);
bigfx_t bigfx_sq(bigfx_t a);
bigfx_t bigfx_sqrt(bigfx_t a);
bigfx_t bigfx_tanh(bigfx_t a);

#endif /* ATOMIC_TYPE_BIGFX_STATE_H_ */
