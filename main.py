import argparse
import keras
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Add
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import Model as M
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model
from model import Model
from path import MODEL_PATH
from keras.callbacks import LearningRateScheduler

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=600, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()

sqeue = ResNet50(include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=(32, 32, 3),
                 pooling=None,
                 classes=10, )

# plot_model(sqeue, 'resxnet.png', show_shapes=True)
# sqeue = model(weights=None, classes=10, input_shape=(32, 32, 3))
# initiate RMSprop optimizer
opt = keras.optimizers.adam(1e-4)

# Let's train the model using RMSprop
loss = 'categorical_crossentropy'
sqeue.compile(loss=[loss],
              optimizer=opt,
              metrics=['accuracy'])
# sqeue.summary()

dataset = Dataset()
model = Model(dataset)
best_score = 0.

from keras.datasets import cifar10
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
BS = 32


def step_decay(epoch):
    import math
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


for epochs in range(args.EPOCHS):
    # x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
    history = sqeue.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
                                  steps_per_epoch=32,
                                  # epochs=args.EPOCHS,
                                  validation_data=(x_test, y_test))
    score = sqeue.evaluate(x_test, y_test, verbose=0)
    if score[-1] > best_score:
        best_score = score[-1]
        model.save_model(sqeue, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (epochs, best_score))
    print(str(epochs) + "/" + str(args.EPOCHS))
