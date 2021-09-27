#include <stdio.h>

#include "zeronn/tensor.h"

int main(void) {
    int size = 1000000;
    tensor_t* x = uniform_log2(bigfx_from_fx(-(1 << 7)), 7 + 1, size);
    for (int i = 0; i < size; ++i) {
        int d = fx_from_bigfx(x->data[i]);
        printf("%d\n", d);
    }

    tensor_free(x);
}
