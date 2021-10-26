#!/bin/sh

wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xvf cifar-10-binary.tar.gz
mkdir -p data/cifar10/
ls cifar-10-batches-bin/data_batch_* | sort | xargs cat > data/cifar10/train.bin
mv cifar-10-batches-bin/test_batch.bin data/cifar10/val.bin
rm cifar-10-binary.tar.gz
rm -rf cifar-10-batches-bin/
