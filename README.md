# advGAN_HR: ImageNet images

This code is taken from https://github.com/niharikajainn/adv_gan_keras and adapted for generating high-resolution adversarial images by using GAN. Unlike the classical method of training the generator with many images, here we train it with only one clean image (our target image) and convert it to the target category for the target CNN. Target CNN is VGG-16 trained on ImageNet dataset (you can use any CNN trained on ImageNet); clean image is one of the acorn images; target category is 306 -  'rhinoceros_beetle'. You can change these values as you wish. The "epochs" value is fixed at 200, but can be increased for different clean/target pairs. If the adversarial image is successfully produced within the "epochs", it will be saved as advers_'epochs'.png.

How to run: After installing required libraries, run main.py.
