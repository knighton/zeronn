#include <stdio.h>

#include "zeronn/prng.h"

int main(void) {
    for (int i = 0; i < 1000; ++i) {
        prng_set_seed(i);
        printf("%d:", i);
        for (int j = 0; j < 1000; ++j) {
            int low = -j;
            int high = low + (1 << (j % 20));
            int a = prng_randint(low, high);
            printf(" %d", a);
        }
        printf("\n");
    }
}
