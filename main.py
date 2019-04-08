import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
from keras.models import model_from_json
import glob 
import os
from tqdm import tqdm 

from train import *


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def load_model():
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss=euclidean_distance_loss, optimizer='Adam')
    return loaded_model

def load_checkpoint(): 
    model = model3() 
    list_of_files = glob.glob('checkpoints/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    model.load_weights(latest_file)
    model.compile(loss=euclidean_distance_loss, optimizer='Adam')
    return model 

def find_circle(img, loaded_model):
    # loaded_model.compile(loss=euclidean_distance_loss, optimizer='Adam')
    y = loaded_model.predict(img)

    return y[0]


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    # model = load_model()
    model = load_checkpoint()
    for _ in tqdm(range(1000)):
        params, img = noisy_circle(200, 50, 2)
        img = img.reshape(1, 200, 200, 1)
        detected = tuple(find_circle(img, model))
        # print("params: {}, detected: {}".format(params, detected))
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

if __name__ == "__main__":
    main()
