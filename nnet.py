import numpy as np
from tqdm import tqdm

class Network:
    @staticmethod
    def generate_weights(m, n):
        # Xavier Weight Initialization
        return np.random.uniform(
            -np.sqrt(6.0 / (n + 1)), np.sqrt(6.0 / (n + 1)), size=(m, n)
            )
    

    @staticmethod
    def cross_entropy(y, y_hat):
        return -1 * np.sum( y * np.log(y_hat))


    def __init__(self, number_layers = 3, layer_size = 250, 
                 output_size = 100, input_size = 1568):
        
        weights = list() 
        self.deltas = list() # Used for gradient 
        biases = list() 

        values = [np.zeros(layer_size) for i in range(number_layers - 2)]
        values.append(np.zeros(output_size))

        weights.append(Network.generate_weights(layer_size, input_size))
        biases.append(np.full(layer_size, 0.01))
       
        # Construction
        for i in range(number_layers - 2):
            n = weights[-1].shape[0]

            # Output Layer   
            if i != number_layers - 1:
                weights.append(Network.generate_weights(output_size, n))
                biases.append( np.full(output_size, 0.1))

            # Hidden Layers
            else:
                weights.append(Network.generate_weights(layer_size, n))
                biases.append(np.full(layer_size, 0.1))

        self.input_array = np.zeros(input_size)
        self.weights = weights
        self.number_layers = number_layers
        self.biases = biases
        self.loss = -1
        self.learn_rate = 0.1
        self.values = [i.reshape(i.size, 1) for i in values]#Reshape to column


    def forwards(self, input_array, output_label = False):
        input_array = input_array.flatten()

        # Functions
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        soft_max = np.vectorize(
                                lambda x: np.exp(x) / np.sum(np.exp(output_z))
                   )

        # z = WA + B
        z = np.vectorize(
                lambda i: self.weights[i] @ self.values[i-1] + self.biases[i]
            )

        # First layer
        self.values[0] = sigmoid(self.weights[0] @ input_array + self.biases[0])

        # Middle Layers
        for i in range(1, len(self.weights) - 1):
            self.values[i] = sigmoid(z(i)) 
        
        # Last Layer
        output_z = z(-1)
        self.values[-1] = soft_max( output_z )
           
        if output_label:
            return np.argmax(self.values[-1] + 1)

        self.values = [i.reshape(i.size, 1) for i in self.values] 
        
        return self.values[-1] 


    def backwards_propogate(self, input_array, label):
        delta = list()

        input_array = input_array.flatten()
        y_hat = self.forwards(input_array).reshape(-1, 1) # Prediction
        y = np.zeros_like(y_hat) # Ground Truth
        y[label] = 1

        self.loss = Network.cross_entropy(y, y_hat)

        # Output Gradient
        delta.append( y_hat - y)
        for i in range(self.number_layers - 2, 0, -1):
            
            # Weight matrix error
            per_error = self.weights[i].T @ delta[-1]

            # Derivative of activation function
            act_grad = self.values[i -1] * (1 - self.values[i -1]) 
            
            # Final gradient 
            delta.append(per_error * act_grad)
        
        delta = delta[::-1] 
        #self.gradient_check(y, input_array, delta)
        self.update_weights(delta)


    def update_weights(self, delta):
        learn_rate = self.learn_rate 
        input_array = self.input_array.reshape(-1, 1)

        for idx, weight in enumerate(self.weights):
            if idx == 0:
                weight -= learn_rate * np.dot(delta[idx], input_array.T)
            else: 
                weight -= learn_rate * np.dot(delta[idx], self.values[idx-1].T)

            self.biases[idx] -=  learn_rate * np.sum( delta[idx].reshape(-1))


    # Gets the loss if we nudge the gradient by e
    def epsilon_forwards(self, input_array, epsilon, lay, nod, wei):
        weights = [np.copy(i) for i in self.weights]
        weights[lay][nod][wei] += epsilon
        
        values = [np.copy(i) for i in self.values]
        biases = [np.copy(i) for i in self.biases]

        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-1 * x)))
        soft_max = lambda x: np.exp(x) / np.sum( np.exp(x) )
        z = lambda i: weights[i] @ values[i - 1] + biases[i]

        input_array = input_array.reshape(-1)

        # First layer
        values[0] = sigmoid(weights[0] @ input_array + biases[0]).reshape(-1, 1)

        # Middle Layers
        for i in range(1, len(self.weights) - 1):
            values[i] = sigmoid(z(i)) 

        # Last Layer
        values[-1] = soft_max(z(-1)) 
        return values[-1]


    def gradient_check(self, y, input_array, delta):
        epsilon = 0.0000001 
        
        lay = 1 
        nod = 0 
        wei = 0

        # Getting the output by +/- e 
        pos = self.epsilon_forwards(input_array, epsilon, lay, nod, wei)
        neg = self.epsilon_forwards(input_array, -1 * epsilon, lay, nod, wei)
       
        l_pos = Network.cross_entropy(y, pos)
        l_neg = Network.cross_entropy(y, neg)

        # Numeric Gradient 
        predic = (l_pos - l_neg) / (2 * epsilon)
        predic = predic / len(self.values[lay])
        input_array = input_array.reshape(-1, 1)

        if lay == 0:
            formulic = np.dot(input_array, delta[0].T).T

        else: 
            formulic = np.dot(self.values[lay - 1], delta[lay].T).T

        formulic = formulic[nod][wei] 
        print('Numeric: ', predic) 
        print('Formulaic: ', formulic)
        print('Error : ', abs(predic - formulic) / max(abs(predic), abs(formulic)))
        breakpoint()


    def train(self, images, labels, n_epoch):
        losses = list()

        for epoch in range(n_epoch):
            print('epoch: ', epoch, '/', n_epoch)
            if epoch != 0:
                print(round(np.mean(losses), 4))
            losses = [] 

            for idx in tqdm(range(len(images))):
                label = labels[idx]
                image =  images[idx]

                self.backwards_propogate(image, label)

                losses.append(self.loss)
                if idx % 1000 == 0:
                    self.rate = self.rate / 1.01
                    #print(round(np.mean(losses), 4))


    def evaluate(self, images, labels):
        right = []
        wrong = []

        for idx, label in enumerate(labels):
            temp = net.forwards( img, True) == label 

            if temp: 
                right.append(label)

            else:
                wrong.append(label)
         
        print('right: ', {i : right.count(i) for i in set(right)})
        print('wrong: ', {i : wrong.count(i) for i in set(wrong)}) 
        print('accuracy: ', len(right) / (len(right) + len(wrong) ))
