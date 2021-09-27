CC = clang

CFLAGS = \
	-O3 \
	-std=c99 \
	-Wall -Werror -Wpedantic -Weverything -Wextra \
	-Wno-missing-prototypes -Wno-conditional-uninitialized -Wno-conversion \
	-Wno-sign-conversion -Wno-padded -Wno-cast-align -Wno-switch-enum \
	-Wno-double-promotion -Wno-covered-switch-default -Wno-unused-macros \
	-Wno-unused-parameter

FP_CC = `find src/fp/zeronn/ -type f -name "*.c"`
FX_CC = `find src/fx/zeronn/ -type f -name "*.c"`
BIGFX_CC = `find src/bigfx/zeronn/ -type f -name "*.c"`

all: fp fx bigfx

fp:
	mkdir -p bin/fp/
	$(CC) src/fp/fit.c $(FP_CC) -Isrc/fp/ -lm -o bin/fp/fit $(CFLAGS)

fx:
	mkdir -p bin/fx/
	$(CC) src/fx/fit.c $(FX_CC) -lm -Isrc/fx/ -o bin/fx/fit $(CFLAGS)
	$(CC) src/fx/gen_prng.c $(FX_CC) -lm -Isrc/fx/ -o bin/fx/gen_prng $(CFLAGS)

bigfx:
	mkdir -p bin/bigfx/
	$(CC) src/bigfx/fit.c $(BIGFX_CC) -lm -Isrc/bigfx/ -o bin/bigfx/fit $(CFLAGS)
	$(CC) src/bigfx/gen_bigfx.c $(BIGFX_CC) -lm -Isrc/bigfx/ -o bin/bigfx/gen_bigfx $(CFLAGS)
	$(CC) src/bigfx/gen_prng.c $(BIGFX_CC) -lm -Isrc/bigfx/ -o bin/bigfx/gen_prng $(CFLAGS)
	$(CC) src/bigfx/gen_uniform.c $(BIGFX_CC) -lm -Isrc/bigfx/ -o bin/bigfx/gen_uniform $(CFLAGS)
	$(CC) src/bigfx/test_bigint.c $(BIGFX_CC) -lm -Isrc/bigfx/ -o bin/bigfx/test_bigint $(CFLAGS)
	$(CC) src/bigfx/test_biguint.c $(BIGFX_CC) -lm -Isrc/bigfx/ -o bin/bigfx/test_biguint $(CFLAGS)

test_bigfx:
	mkdir -p data/bigfx_plot/
	./bin/bigfx/gen_bigfx > data/bigfx.txt
	python3 src/bigfx/plot_bigfx.py data/bigfx.txt data/bigfx_plot/

fit:
	python3 -m zeronn.orig.fit
	@echo
	python3 -m zeronn.auto.fit
	@echo
	python3 -m zeronn.fp.fit
	@echo
	python3 -m zeronn.fx.fit
	@echo
	./bin/fp/fit
	@echo
	./bin/fx/fit
	@echo
	./bin/bigfx/fit
