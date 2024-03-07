import numpy as np
from random import randint
from nnet import Network
import matplotlib.pyplot as plt
import csv

def plot_losses(losses):
    # Loss Plot
    xs = [x for x in range(len(losses))]
    plt.plot(xs, losses)
    plt.savefig(f'losses.png')

def grid_search():
    data = r'/Users/work/Data/mnist'

    train_images = np.load(data + r'/train_images.npy') / 255 
    train_labels = np.load(data + r'/train_labels.npy') 

    test_images = np.load(data + r'/test_images.npy') / 255
    test_labels = np.load(data + r'/test_labels.npy') 
  

    test_pair = lambda i: (train_images[i], train_labels[i])

    test = train_images[0]


    for layers in range(3, 5):
        for layer_size in [500, 750, 1000]:
                
            net = Network(number_layers = layers, layer_size = layer_size)
           

            losses = []
            right = []
            wrong = []

            
            net.train(train_images, train_labels, 1)

            # Accuracy 
            for idx, img in enumerate(test_images):
                temp = net.forwards( img, True) == test_labels[idx]

                if temp: 
                    right.append(test_labels[idx])

                else:
                    wrong.append(test_labels[idx])
            
            percent = len(right) / (len(wrong) + len(right))
            with open('meta.csv', 'a') as file:
                file.write(f'{layers}, {layer_size}, {percent} \n')

            print('-' * 100, '\n \n')


if __name__ == '__main__':
    grid_search()

