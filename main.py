from __future__ import print_function
import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.backend import clip

from InstanceNormalization import InstanceNormalization

import os, cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed


class DCGAN():

    def __init__(self):
        # input image dimensions
        self.img_width = 224
        self.img_height = 224
        self.input_shape = (self.img_width, self.img_height, 3)  # 3 channels here for VGG19

        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        # Build generator
        inputs = Input(shape=self.input_shape)
        outputs = self.build_generator(inputs)
        self.G = Model(inputs, outputs)
        self.G._name = 'Generator'
        # self.G.summary()

        # Build discriminator and train it
        outputs = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, outputs)
        self.D.compile(loss=tensorflow.keras.losses.binary_crossentropy, optimizer=optimizer_d,
                       metrics=[self.custom_acc])
        self.D._name = 'Discriminator'
        # self.D.summary()

        # We use VGG16 trained with ImageNet dataset.
        self.target = VGG16(weights='imagenet')
        self.target.trainable = False

        # Build GAN: stack generator, discriminator and target
        img = (self.G(inputs) / 2 + 0.5) * 255  # image's pixels will be between [0, 255]
        self.stacked = Model(inputs=inputs,
                             outputs=[self.G(inputs), self.D(self.G(inputs)), self.target(preprocess_input(img))])
        self.stacked.compile(loss=[self.generator_loss, tensorflow.keras.losses.binary_crossentropy,
                                   tensorflow.keras.losses.categorical_crossentropy], optimizer=optimizer_g)
        self.stacked.summary()

    def generator_loss(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)  # Hinge loss

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_pred), K.round(y_true))

    def build_discriminator(self, inputs):
        D = Conv2D(32, 4, strides=(2, 2))(inputs)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Conv2D(64, 4, strides=(2, 2))(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)
        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D

    def build_generator(self, inputs):
        # c3s1-8
        G = Conv2D(8, 3, padding='same')(inputs)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        # d16
        G = Conv2D(16, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        # d32
        G = Conv2D(32, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)
        residual = G
        
        # four r32 blocks
        for _ in range(4):
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G
        # u16
        G = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)
        # u8
        G = Conv2DTranspose(8, 3, strides=(2, 2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        G = Conv2D(1, 3, padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)
        G = layers.add([G*2.25/255, inputs])  
        # Multiplying 2.25/255 = 0.01 will drastically reduce the magnitude of the noise, making it invisible. You can increase                                                 # this value, which would help advGAN_HR to generate the adversarial image more easily, but the visual quality will decrease. 
        return G



    def train_D_on_batch(self, x_batch, Gx_batch, y_batches):
        # x_batch, Gx_batch, y_batches = batches
        # for each batch:
        # predict noise on generator: G(z) = batch of fake images
        # train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        # train real images on disciminator: D(x) = update D params per classification for real images
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(
        len(x_batch), 1)))  # real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)))  # fake=0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss  # (loss, accuracy) tuple

    def train_stacked_on_batch(self, x_batch, _, y_batch):
        # x_batch, _, y_batch = batches
        arr = np.zeros(1000)
        arr[targx] = 1  # tarx is index of the target class
        full_target = np.tile(arr, (len(x_batch), 1))

        # for each batch:
        # train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        # Update only G params
        self.D.trainable = False
        self.target.trainable = False

        # input to full GAN is original image
        # output 1 label for generated image is original image
        # output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
        # output 3 label for target classification is 724; g wants to generate pirate images
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), full_target])
        # outputs = [self.G(inputs), self.D(self.G(inputs)), self.target(self.G(inputs))]
        return stacked_loss  # (total loss, hinge loss, gan loss, adv loss) tuple

    def prepare_data(self):
        dataset = []
        path = 'ancestors/abacus/'
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # testing
                dataset.append(cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_NEAREST))

        dataset = np.array(dataset, dtype=np.float32)
        dataset = (dataset * 2. / 255 - 1).reshape(len(dataset), 224, 224, 3)  # normalize images to [-1, 1]
        return dataset, np.full(1, anc)  # anc is the index of clean image class

    def train_GAN(self):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_train = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
        x_train = np.expand_dims(x_train, axis=0)
        x_train = np.array(x_train, dtype=np.float32)
        x_train = (x_train * 2. / 255 - 1).reshape(len(x_train), 224, 224, 3)  # normalize images to [-1, 1]
        y_train = np.full(1, anc)

        epochs = 200
        for epoch in range(epochs):
            print("EPOCH: ", epoch)
            Gx = self.G.predict(x_train)
            (d_loss, d_acc) = self.train_D_on_batch(x_train, Gx, y_train)
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_stacked_on_batch(x_train, Gx, y_train)

            print("===========================================")
            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: %f"
                  % (d_loss, d_acc * 100., gan_loss, hinge_loss, adv_loss))

            np.save('adversImg.npy',Gx)

            # report each epoch the adversarial image classification; dominant class, target class probabilities.
            img_normalized = np.load("adversImg.npy").copy()
            img = (img_normalized / 2.0 + 0.5) * 255
            image = img.reshape((1, 224, 224, 3))
            yhat = self.target.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            label1 = label0[0][0]
            for aa in label0[0]:
                if aa[1] == target:
                    print(aa[1], " ", aa[2])
            print(label1[1], label1[2])
            print("==============================================")
            if np.argmax(yhat, axis=1) == targx or epoch == epochs-1:
                img = (img_normalized / 2.0 + 0.5) * 255           
                Img = Image.fromarray((img[0]).astype(np.uint8))
                filename = f"advers_{epoch}.png"
                Img.save(filename, 'png')
                break


path = "acorn2.JPEG"  # clean image
target = 'rhinoceros_beetle'  # target category label
targx = 306   # target category index no
anc = 988    # clean image index no

if __name__ == '__main__':
    seed(5)
    tensorflow.random.set_seed(1)
    dcgan = DCGAN()
    dcgan.train_GAN()



# pairs = {'abacus':[398, 421, 'bannister'],'acorn':[988,306,'rhinoceros_beetle'],'baseball':[429,618, 'ladle'],
#                        'brown_bear':[294,724, 'pirate'],'broom':[462,273, 'dingo'],'canoe':[472,176, 'Saluki'],
#                        'hippopotamus':[344,927,'trifle'],'llama':[355,42,'agama'],'maraca':[641,112,'conch'],'mountain_bike':[671, 828,'strainer']}M
