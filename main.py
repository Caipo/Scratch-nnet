import numpy as np
from random import randint
from nnet import Network
from aux import plot_losses
import threading

if __name__ == '__main__':
    data = r'/Users/work/Data/mnist'

    train_images = np.load(data + r'/train_images.npy') / 255 
    train_labels = np.load(data + r'/train_labels.npy') 

    test_images = np.load(data + r'/test_images.npy') / 255
    test_labels = np.load(data + r'/test_labels.npy') 
   

    test_pair = lambda i: (train_images[i], train_labels[i])

    test = train_images[0]
    net = Network(4, 1500)

    losses = net.train(train_images, train_labels, 5)
    plot_losses(losses)
    net.evaluate(test_images, test_labels)
    breakpoint()
    exit()
