import numpy as np
from random import randint
from nnet import Network
import threading

if __name__ == '__main__':
    data = r'/Users/work/Data/mnist'

    train_images = np.load(data + r'/train_images.npy') / 255 
    train_labels = np.load(data + r'/train_labels.npy') 

    test_images = np.load(data + r'/test_images.npy') / 255
    test_labels = np.load(data + r'/test_labels.npy') 
   

    test_pair = lambda i: (train_images[i], train_labels[i])

    test = train_images[0]
    net = Network()
   

    losses = []
    right = []
    wrong = []

    net.train(train_images, train_labels, 5)

    # Accuracy 
    for idx, img in enumerate(test_images):
        temp = net.forwards( img, True) == test_labels[idx]

        if temp: 
            right.append(test_labels[idx])

        else:
            wrong.append(test_labels[idx])
        
        
    net.evaluate(test_images, test_labels)
    breakpoint()
