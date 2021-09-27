#ifndef ZERONN_TYPE_BIG_CONV_H_
#define ZERONN_TYPE_BIG_CONV_H_

#include "zeronn/type/bigint.h"
#include "zeronn/type/biguint.h"

bigint_t bigint_from_biguint(biguint_t x);
biguint_t biguint_from_bigint(bigint_t x);

#endif /* ZERONN_TYPE_BIG_CONV_H_ */
