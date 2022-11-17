import numpy as np
import cv2
import numpy.core.multiarray
import pickle
import os
from imageio import imread
from tqdm import tqdm

def extract_features(image_path, vector_size = 256):
    image = imread(image_path, mode = "RGB")
    try:
        kaze = cv2.KAZE_create()
        fiture = kaze.detect(image)
        fiture = sorted(fiture, key = lambda x: -x.response)[:vector_size]
        fiture, vector = kaze.compute(image, fiture)
        vector = vector.flatten()
        needed_size = (vector_size * 512)
        if vector.size < needed_size:
            vector = np.concatenate([vector, np.zeros(needed_size - vector.size)])
    except cv2.error as e:
        print("Error: ", e)
        return None

    return vector

result = {}

def batch_extractor(images_path):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    for f in tqdm(files):
        name = f.split("\\")[-1].lower()
        result[name] = extract_features(f)

def batch_dump(path):
    with open(path, "wb") as fp:
        pickle.dump(result, fp)

def run(resource_path=r'data wajah'):
    batch_extractor(resource_path)
    batch_dump("features.pck")
