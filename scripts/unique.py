# using in tf
from os import dup
import os
from imagededup.methods import PHash

# PHash 162
# DHash 140
# WHash 64
# AHash 52
# CNN 92

phasher = PHash()
dt = {}

for folder in os.listdir('/home/djy/dataset/dataset2'):
    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(
        image_dir=os.path.join('/home/djy/dataset/dataset2', folder))

    # Find duplicates using the generated encodings
    duplicates: dict = phasher.find_duplicates(encoding_map=encodings)

    only = []
    dups = []

    for k, v in duplicates.items():
        # print(dup, len(duplicates[dup]))
        f = False
        for item in v:
            f |= (k in duplicates[item])
        if not f:
            only.append(k)
    dt[folder] = len(only)
    for file in os.listdir(os.path.join('/home/djy/dataset/dataset2', folder)):
        if file not in only:
            os.remove(os.path.join('/home/djy/dataset/dataset2', folder, file))
