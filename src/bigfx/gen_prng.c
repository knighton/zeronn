#include <stdint.h>
#include <stdio.h>

#include "zeronn/prng.h"
#include "zeronn/type/bigint.h"
#include "zeronn/type/biguint.h"

int main(void) {
    for (uint32_t i = 0; i < 1000; ++i) {
        biguint_t seed = biguint_from_uint(i);
        prng_set_seed(seed);
        printf("%u:", i);
        for (int j = 0; j < 1000; ++j) {
            bigint_t low = bigint_from_int(-j);
            uint8_t range_log2 = j % 20;
            bigint_t a2 = prng_randint_log2(low, range_log2);
            int a = int_from_bigint(a2);
            printf(" %d", a);
        }
        printf("\n");
    }
}
