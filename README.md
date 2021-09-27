# zeronn

Neural network demo, as a series of increasingly awesome rewrites from Pytorch to C to Brainfuck:

1. Vanilla pytorch [`zeronn/orig/`](zeronn/orig/)
2. No modules, provide forward pass, use autograd [`zeronn/auto/`](zeronn/auto/)
3. Autograd banned, provide backward pass, like numpy [`zeronn/fp/`](zeronn/fp/)
4. Floats banned, use integer fixed point [`zeronn/fx/`](zeronn/fx)
5. Rewrite #3 in C99 [`src/fp/`](src/fp)
6. Rewrite #4 in C99 [`src/fx/`](src/fx)
7. Ints banned, use byte bignum fixed point, in C99 [`src/bigfx/`](src/bigfx)
