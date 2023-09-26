"""
make_imagenet_subset.py

Test utility for making small subsets of the ILSVRC database;
creates a directory at the specified target path containing
symlinks to folders full of training images
"""

import os
import random

# Make the enclosing directory
target_path = os.path.join(os.getcwd(), 'imagenet_subset')
os.mkdir(target_path)

# Extract ILSVRC subset
imagenet_path = os.path.join(os.getcwd(), 'ImageNet/ILSVRC/Data/CLS-LOC/train/') 
all_paths = [y for y in [os.path.join(imagenet_path, x) for x in os.listdir(imagenet_path)] if os.path.isdir(y)]
subset_paths = random.sample(all_paths, 10)

# Make symlinks
symlink_paths = [os.path.join(target_path, os.path.basename(x)) for x in subset_paths]

for src, dest in zip(subset_paths, symlink_paths):
    #print(f"{src} -> {dest}")
    os.symlink(src, dest)
