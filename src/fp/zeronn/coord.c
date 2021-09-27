#include "coord.h"

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>

int pp_coord(int num_args, ...) {
    int x = 0;
    va_list args;
    va_start(args, num_args);
    if (num_args == 1) {
        int w = va_arg(args, int) % 256;
        x = w;
    } else if (num_args == 2) {
        int h = va_arg(args, int) % 256;
        int w = va_arg(args, int) % 256;
        x = (h << 8) + w;
    } else if (num_args == 3) {
        int d = va_arg(args, int) % 256;
        int h = va_arg(args, int) % 256;
        int w = va_arg(args, int) % 256;
        x = (d << 16) + (h << 8) + w;
    } else {
        assert(false);
    }
    va_end(args);
    return x;
}

void coord4d_init(int x, coord4d_t* coord, int min) {
    if (x < 256) {
        assert(min <= x);
        coord->time = x;
        coord->depth = x;
        coord->height = x;
        coord->width = x;
        return;
    }

    coord->width = x % 256;
    x /= 256;
    coord->height = x % 256;
    x /= 256;
    coord->depth = x % 256;
    x /= 256;
    coord->time = x % 256;

    assert(min <= coord->width);
    assert(min <= coord->height);
    assert(min <= coord->depth);
    assert(min <= coord->time);
}

void coord3d_init(int x, coord3d_t* coord, int min) {
    if (x < 256) {
        assert(min <= x);
        coord->depth = x;
        coord->height = x;
        coord->width = x;
        return;
    }

    coord->width = x % 256;
    x /= 256;
    coord->height = x % 256;
    x /= 256;
    coord->depth = x % 256;

    assert(min <= coord->width);
    assert(min <= coord->height);
    assert(min <= coord->depth);
}

void coord2d_init(int x, coord2d_t* coord, int min) {
    if (x < 256) {
        assert(min <= x);
        coord->height = x;
        coord->width = x;
        return;
    }

    coord->width = x % 256;
    x /= 256;
    coord->height = x % 256;

    assert(min <= coord->width);
    assert(min <= coord->height);
}

void coord1d_init(int x, coord1d_t* coord, int min) {
    coord->width = x % 256;

    assert(min <= coord->width);
}
