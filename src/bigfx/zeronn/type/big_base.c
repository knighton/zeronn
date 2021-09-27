#include "big_base.h"

void mul_8_8_16(uint8_t a, uint8_t b, uint8_t* c_hi, uint8_t* c_lo) {
    uint8_t md1;
    uint8_t md2;
    uint8_t old;

    *c_hi = (a >> 4) * (b >> 4);
    md1 = (a >> 4) * (b & 0xF);
    md2 = (a & 0xF) * (b >> 4);
    *c_lo = (a & 0xF) * (b & 0xF);

    *c_hi += md1 >> 4;
    *c_hi += md2 >> 4;

    old = *c_lo;
    *c_lo += (md1 << 4);
    if (*c_lo < old) {
        *c_hi += 1;
    }

    old = *c_lo;
    *c_lo += (md2 << 4);
    if (*c_lo < old) {
        *c_hi += 1;
    }
}

void shift_left_le8(uint8_t n, uint8_t *u, uint8_t m) {
    uint8_t k;
    uint8_t t;
    uint8_t i;

    if (!m) {
        return;
    }

    k = 0;
    for (i = n - 1; i < n; i--) {
        t = u[i] >> (8 - m);
        u[i] = (u[i] << m) | k;
        k = t;
    }
}

void shift_left(uint8_t n, uint8_t *u, uint8_t m) {
    for (uint8_t i = 0; i < m / 8; ++i) {
        shift_left_le8(n, u, 8);
    }
    shift_left_le8(n, u, m % 8);
}

void shift_right_le8(uint8_t n, uint8_t *u, uint8_t m) {
    uint8_t k;
    uint8_t t;
    uint8_t i;

    if (!m) {
        return;
    }

    k = 0;
    for (i = 0; i < n; i++) {
        t = u[i] << (8 - m);
        u[i] = (u[i] >> m) | k;
        k = t;
    }
}

void shift_right(uint8_t n, uint8_t *u, uint8_t m) {
    for (uint8_t i = 0; i < m / 8; ++i) {
        shift_right_le8(n, u, 8);
    }
    shift_right_le8(n, u, m % 8);
}
