import numpy as np
from tqdm import tqdm

class Network:
    @staticmethod
    def generate_weights(m, n):
        return np.random.uniform(
            -np.sqrt(6.0 / (n + 1)), np.sqrt(6.0 / (n + 1)), size=(m, n)
            )
    
    @staticmethod
    def cross_entropy(y, y_hat):
        return -1 * np.sum( y * np.log(y_hat))


    def __init__(self, number_layers = 3, layer_size = 250, output_size = 100, 
                input_size = 1568):
        
        weights = list() 
        self.deltas = list() 
        biases = list() 

        values = [np.zeros(layer_size) for i in range(number_layers - 2)]
        values.append(np.zeros(output_size))

        weights.append(Network.generate_weights(layer_size, input_size))
        biases.append(np.full(layer_size, 0.01))
       
        for i in range(number_layers - 2):
            n = weights[-1].shape[0]

            # Output Layer   
            if i != number_layers - 1:
                weights.append(Network.generate_weights(output_size, n))
                biases.append( np.full(output_size, 0.1))

            # Hidden Layers
            else:
                weights.append(Network.generate_weights(layer_size, n))
                biases.append( np.full(layer_size, 0.1))
        
        self.input_arry = np.zeros(input_size)
        self.weights = weights
        self.number_layers = number_layers
        self.biases = biases
        self.loss = -1
        self.rate = 0.1
        self.values = [i.reshape(i.size, 1) for i in values] 
    
    def forwards(self, input_arry, output_label = False):
        input_arry = input_arry.flatten()

        # Functions
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        soft_max = np.vectorize(lambda x: np.exp(x) / np.sum( np.exp(output_z)))
        z = np.vectorize(lambda i: self.weights[i] @ self.values[i-1] + self.biases[i])

        # First layer
        self.values[0] = sigmoid(self.weights[0] @ input_arry + self.biases[0])

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


    def backwards_propogate(self, input_arry, label):
        delta = list()

        input_arry = input_arry.flatten()
        y_hat = self.forwards(input_arry).reshape(-1, 1)
       
        y = np.zeros_like(y_hat)
        y[label] = 1

        self.loss = Network.cross_entropy(y, y_hat)

        # Output gradiant
        delta.append( y_hat - y)
        for i in range(self.number_layers - 2, 0, -1):
            per_error = self.weights[i].T @ delta[-1]
            act_grad = self.values[i -1] * (1 - self.values[i -1]) 

            delta.append(per_error * act_grad)
        
        delta = delta[::-1] 
        
        #self.gradiant_check(y, input_arry, delta)
        self.update_weights(delta)

    def update_weights(self, delta):
        rate = self.rate 
        input_arry = self.input_arry.reshape(-1, 1)

        for idx, weight in  enumerate(self.weights):

            if idx == 0:
                weight -= rate * np.dot(delta[idx], input_arry.T)
            else: 
                weight -= rate * np.dot(delta[idx], self.values[idx-1].T)

            self.biases[idx] -=  rate * np.sum( delta[idx].reshape(-1))



    def epsilon_forwards(self, input_arry, epsilon, lay, nod, wei):

        weights = [np.copy(i) for i in self.weights]
        weights[lay][nod][wei] += epsilon
        
        values = [np.copy(i) for i in self.values]
        biases = [np.copy(i) for i in self.biases]
        #biases[lay][nod] += epsilon 

        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-1 * x)))
        soft_max = lambda x: np.exp(x) / np.sum( np.exp(x) )
        z = lambda i: weights[i] @ values[i - 1] + biases[i]

        input_arry = input_arry.reshape(-1)

        # First layer
        values[0] = sigmoid(weights[0] @ input_arry + biases[0]).reshape(-1, 1)

        # Middle Layers
        for i in range(1, len(self.weights) - 1):
            values[i] = sigmoid(z(i)) 

        # Last Layer
        values[-1] = soft_max(z(-1)) 
           
        return values[-1]



    def gradiant_check(self, y, input_arry, delta):
        epsilon = 0.0000001 
        
        lay = 1 
        nod = 0 
        wei = 0

        pos = self.epsilon_forwards(input_arry, epsilon, lay, nod, wei)
        neg = self.epsilon_forwards(input_arry, -1 * epsilon, lay, nod, wei)

       
        l_pos = Network.cross_entropy(y, pos)
        l_neg = Network.cross_entropy(y, neg)

        predic = (l_pos - l_neg) / (2 * epsilon)
        predic = predic / len(self.values[lay])
        input_arry = input_arry.reshape(-1, 1)

        if lay == 0:
            current = np.dot( input_arry, delta[0].T).T

        else: 
            current = np.dot( self.values[lay - 1], delta[lay].T).T

        current =  current[nod][wei] 
        print('Numeric: ', predic) 
        print('Formulaic: ', current)
        print('Error : ', abs(predic - current) / max(abs(predic), abs(current)))

        breakpoint()


    def train(self, images, labels, n_epoc):
        losses = list()

        for epoc in range(n_epoc):
            print('epoc: ', epoc, '/', n_epoc )
            if epoc != 0:
                print(round(np.mean(losses), 4))
            losses = [] 

            for idx in tqdm(range(len(images))):
                lable = labels[idx]
                imgs =  images[idx]

                self.backwards_propogate(imgs, lable)

                losses.append(self.loss)
                if idx % 1000 == 0:
                    self.rate = self.rate / 1.01
                    #print(round(np.mean(losses), 4))


    def evaluate(self, images, labels):
        
        right = []
        wrong = []

        for idx, lable in enumerate(labels):
            temp = net.forwards( img, True) == lable 

            if temp: 
                right.append(lable)

            else:
                wrong.append(lable)
        
         
        print('right: ', {i : right.count(i) for i in set(right)})
        print('wrong: ', {i : wrong.count(i) for i in set(wrong)}) 
        print('accuracy: ', len(right) / (len(right) + len(wrong) ))
