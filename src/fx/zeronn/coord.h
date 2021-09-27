#ifndef ZERONN_COORD_H_
#define ZERONN_COORD_H_

#include <stdint.h>

#include "zeronn/preproc.h"

typedef struct {
    int time;
    int depth;
    int height;
    int width;
} coord4d_t;

typedef struct {
    int depth;
    int height;
    int width;
} coord3d_t;

typedef struct {
    int height;
    int width;
} coord2d_t;

typedef struct {
    int width;
} coord1d_t;

int pp_coord(int num_args, ...);
#define coord(...) pp_coord(PP_NARG(__VA_ARGS__), __VA_ARGS__)

void coord4d_init(int x, coord4d_t* coord, int min);
void coord3d_init(int x, coord3d_t* coord, int min);
void coord2d_init(int x, coord2d_t* coord, int min);
void coord1d_init(int x, coord1d_t* coord, int min);

#endif  /* ZERONN_COORD_H_ */
