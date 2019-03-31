from main import *
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers import ReLU, Dropout
import keras.backend as K
from keras.models import model_from_json

np.random.seed(7)

def relu6(x):
    return K.relu(x, max_value=6)


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def circleModel():
    model = Sequential()
    model.add(Conv2D(24, (7, 7),
                     input_shape=(200,200,1),
                     strides=(2, 2),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv1'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))
    model.add(ReLU())
    model.add(Conv2D(36, (5, 5),
                     strides=(2, 2),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv2'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Dense(500))
    model.add(ReLU())

    model.add(Dense(3))

    return model

def model2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(200,200,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))

    return model

def build_data():
    numSamples = 1000

    size = 200
    rad = 50
    noise = 2

    trainX = []
    trainY = []

    for _ in range(numSamples):
        params, img = noisy_circle(size, rad, noise)
        trainX.append(img)
        trainY.append(params)

    trainX = np.array(trainX, dtype=np.float64)
    trainX = trainX.reshape(trainX.shape[0], size, size, 1)
    trainY = np.array(trainY, dtype=np.uint8)

    return trainX, trainY

def train():
    trainX, trainY = build_data()
    # model = circleModel()
    model = model2()
    model.compile(loss=euclidean_distance_loss, optimizer='Adam')
    model.fit(x=trainX, y=trainY, batch_size=25, epochs=5, verbose=1, validation_split=0.1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    train()