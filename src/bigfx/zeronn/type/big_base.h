#ifndef ZERONN_TYPE_BIG_BASE_H_
#define ZERONN_TYPE_BIG_BASE_H_

#include <stdint.h>

void mul_8_8_16(uint8_t a, uint8_t b, uint8_t* c_hi, uint8_t* c_lo);
void shift_left_le8(uint8_t n, uint8_t *u, uint8_t m);
void shift_left(uint8_t n, uint8_t *u, uint8_t m);
void shift_right_le8(uint8_t n, uint8_t *u, uint8_t m);
void shift_right(uint8_t n, uint8_t *u, uint8_t m);

#endif /* ZERONN_TYPE_BIG_BASE_H_ */
