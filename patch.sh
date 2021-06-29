#!/bin/sh

echo "Patching framework"
echo "Copying file coco.py"
cp /patches/coco.py /tensorpack/examples/FasterRCNN/dataset/coco.py

echo "Copying file train.py"
cp /patches/train.py /tensorpack/examples/FasterRCNN/train.py
