#ifndef ZERONN_PRNG_H_
#define ZERONN_PRNG_H_

#include <stdint.h>

#define ZERONN_RAND_MAX (1u << 31)

typedef struct {
    uint32_t a;
} xorshift32_t;

void xorshift32_free(xorshift32_t* state);
void xorshift32_set_seed(xorshift32_t* state, uint32_t seed);
uint32_t xorshift32_get(xorshift32_t* state);

void prng_set_seed(int seed);
int prng_get(void);
int prng_randint(int low, int high);

#endif  /* ZERONN_PRNG_H_ */
