import keras
import numpy as np
from keras import optimizers
from keras.datasets import mnist
from numbers import Number
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Lambda
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from onepixel.networks.train_plot import PlotLearning

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class CNNMNIST:
    def __init__(self, epochs=200, batch_size=128, load_weights=True, p_fail = 0.1):
        self.name               = 'lenet'
        self.model_filename     = 'networks/models/lenet.h5'
        self.num_classes        = 10
        self.input_shape        = 28, 28, 1
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 391
        self.weight_decay       = 0.0001
        self.log_filepath       = r'networks/models/lenet/'
        self.p_fail  		= p_fail

        if load_weights:
            self._model = load_model(self.model_filename)
    
    def count_params(self):
        return self._model.count_params()

    def build_model(self):
        def IndependentCrashes(p_fail, input_shape = None):
            """ Make dropout without scaling """
            assert isinstance(p_fail, Number), "pfail must be a number"
            return Lambda(lambda x: Dropout(p_fail)(x) * (1 - p_fail), input_shape = input_shape, name = 'Crashes')

        model = Sequential()
        model.add(IndependentCrashes(self.p_fail, input_shape = self.input_shape))
        model.add(Conv2D(8, (3, 3), padding='valid', activation = 'relu', kernel_initializer='random_normal', input_shape=self.input_shape))
        model.add(Conv2D(16, (3, 3), padding='valid', activation = 'relu', kernel_initializer='random_normal'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(50, activation = 'relu', kernel_initializer='he_normal' ))
        model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal' ))
        sgd = optimizers.Adadelta()
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def scheduler(self, epoch):
        if epoch <= 60:
            return 0.05
        if epoch <= 120:
            return 0.01
        if epoch <= 160:    
            return 0.002
        return 0.0004

    def train(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.reshape(x_train, (60000, 28,28,1))
        x_test = np.reshape(x_test, (10000, 28,28,1))

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing

        # build network
        model = self.build_model()
        model.summary()

        # Save the best model during each training checkpoint
        checkpoint = ModelCheckpoint(self.model_filename,
                                    monitor='val_loss', 
                                    verbose=0,
                                    save_best_only= True,
                                    mode='auto')
        plot_callback = PlotLearning()
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)

        cbks = [checkpoint, plot_callback, tb_cb]

        # using real-time data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

        datagen.fit(x_train)

        # start traing 
        model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
                            steps_per_epoch=self.iterations,
                            epochs=self.epochs,
                            callbacks=cbks,
                            validation_data=(x_test, y_test))
        # save model
        model.save(self.model_filename)

        self._model = model

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = img
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(np.array([img]))[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]



