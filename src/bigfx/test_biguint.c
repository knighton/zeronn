#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "zeronn/type/biguint.h"

void test_uint(uint32_t a) {
    biguint_t a2 = biguint_from_uint(a);
    uint32_t a2a = uint_from_biguint(a2);
    assert(a == a2a);
}

void test_mul_log2(uint32_t a) {
    for (uint8_t i = 0; i < 4; ++i) {
        uint32_t b = a << i;
        biguint_t a2 = biguint_from_uint(a);
        a2 = biguint_mul_log2(a2, i);
        uint32_t a2a = uint_from_biguint(a2);
        assert(b == a2a);
    }
}

void test_div_log2(uint32_t a) {
    for (uint8_t i = 0; i < 4; ++i) {
        uint32_t b = a / (1 << i);
        biguint_t a2 = biguint_from_uint(a);
        a2 = biguint_div_log2(a2, i);
        uint32_t a2a = uint_from_biguint(a2);
        assert(b == a2a);
    }
}

void test_mod_log2(uint32_t a) {
    for (uint8_t i = 0; i < 4; ++i) {
        uint32_t b = a % (1 << i);
        biguint_t a2 = biguint_from_uint(a);
        a2 = biguint_mod_log2(a2, i);
        uint32_t a2a = uint_from_biguint(a2);
        assert(b == a2a);
    }
}

void test_one(uint32_t a) {
    test_uint(a);
    test_mul_log2(a);
    test_div_log2(a);
    test_mod_log2(a);
}

void test_add(uint32_t a, uint32_t b) {
    uint32_t c = a + b;
    biguint_t a2 = biguint_from_uint(a);
    biguint_t b2 = biguint_from_uint(b);
    biguint_t c2 = biguint_add(a2, b2);
    uint32_t c2c = uint_from_biguint(c2);
    assert(c == c2c);
}

void test_sub(uint32_t a, uint32_t b) {
    uint32_t c = a - b;
    biguint_t a2 = biguint_from_uint(a);
    biguint_t b2 = biguint_from_uint(b);
    biguint_t c2 = biguint_sub(a2, b2);
    uint32_t c2c = uint_from_biguint(c2);
    assert(c == c2c);
}

void test_mul(uint32_t a, uint32_t b) {
    uint32_t c = a * b;
    biguint_t a2 = biguint_from_uint(a);
    biguint_t b2 = biguint_from_uint(b);
    biguint_t c2 = biguint_mul(a2, b2);
    uint32_t c2c = uint_from_biguint(c2);
    assert(c == c2c);
}

void test_two(uint32_t a, uint32_t b) {
    test_add(a, b);
    test_sub(a, b);
    test_mul(a, b);
}

int main(void) {
    srand(31337);

    for (uint32_t i = -300; i <= 300; ++i) {
        test_one(i);
        for (uint32_t j = -300; j <= 300; ++j) {
            test_two(i, j);
        }
    }

    for (uint32_t i = 0; i < 100000; ++i) {
        uint32_t a = (uint32_t)rand() % 4096;
        uint32_t b = (uint32_t)rand() % 4096;
        test_one(a);
        test_one(b);
        test_two(a, b);
    }
}
