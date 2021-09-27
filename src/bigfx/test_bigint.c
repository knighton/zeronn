#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "zeronn/type/bigint.h"

void test_int(int a) {
    bigint_t a2 = bigint_from_int(a);
    int a2a = int_from_bigint(a2);
    assert(a == a2a);
}

void test_mul_log2(int a) {
    for (uint8_t i = 0; i < 4; ++i) {
        int b = a << i;
        bigint_t a2 = bigint_from_int(a);
        a2 = bigint_mul_log2(a2, i);
        int a2a = int_from_bigint(a2);
        assert(b == a2a);
    }
}

void test_div_log2(int a) {
    for (uint8_t i = 0; i < 4; ++i) {
        int b = a / (1 << i);
        bigint_t a2 = bigint_from_int(a);
        a2 = bigint_div_log2(a2, i);
        int a2a = int_from_bigint(a2);
        assert(b == a2a);
    }
}

void test_one(int a) {
    test_int(a);
    test_mul_log2(a);
    test_div_log2(a);
}

void test_add(int a, int b) {
    int c = a + b;
    bigint_t a2 = bigint_from_int(a);
    bigint_t b2 = bigint_from_int(b);
    bigint_t c2 = bigint_add(a2, b2);
    int c2c = int_from_bigint(c2);
    assert(c == c2c);
}

void test_sub(int a, int b) {
    int c = a - b;
    bigint_t a2 = bigint_from_int(a);
    bigint_t b2 = bigint_from_int(b);
    bigint_t c2 = bigint_sub(a2, b2);
    int c2c = int_from_bigint(c2);
    assert(c == c2c);
}

void test_mul(int a, int b) {
    int c = a * b;
    bigint_t a2 = bigint_from_int(a);
    bigint_t b2 = bigint_from_int(b);
    bigint_t c2 = bigint_mul(a2, b2);
    int c2c = int_from_bigint(c2);
    assert(c == c2c);
}

void test_two(int a, int b) {
    test_add(a, b);
    test_sub(a, b);
    test_mul(a, b);
}

int main(void) {
    srand(31337);

    for (int i = -300; i <= 300; ++i) {
        test_one(i);
        for (int j = -300; j <= 300; ++j) {
            test_two(i, j);
        }
    }

    for (int i = 0; i < 100000; ++i) {
        int a = rand() % 4096 - 2048;
        int b = rand() % 4096 - 2048;
        test_one(a);
        test_one(b);
        test_two(a, b);
    }
}
