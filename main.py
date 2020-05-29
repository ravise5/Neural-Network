import numpy
import NN
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x = pd.read_csv("housepricedata.csv")
columns = x.columns.tolist()
cols_to_use_input = columns[:len(columns)-1]
cols_to_use_output = columns[len(columns)-1:]

x = pd.read_csv("housepricedata.csv", usecols = cols_to_use_input)
y = pd.read_csv("housepricedata.csv", usecols = cols_to_use_output)

x = x.to_numpy()
y = y.to_numpy()
################   INPUT PARAMS   ###################
no_of_layers = 3
no_of_nodes = [10,5,1]
data_scaler = 'min-max'
activations = ['relu','sigmoid']
no_of_iters = 50000
size_of_batch = 1168
learning_rate = 0.0001
dropout_rate = 0.3
plot = False
#####################################################

if data_scaler == 'standardization':
    x = NN.standardization(x)
    print(x)
else:
    x = NN.minmax(x)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

##Creating the initial population.
weights = []
bias = []

for i in numpy.arange(0, no_of_layers-1):
    #numpy.random.seed(0)
    layera = no_of_nodes[i]
    layerb = no_of_nodes[i+1]
    weight_layer = numpy.random.randn(layerb,layera)/numpy.sqrt(layera) #Xavier Initialization - Make your Own Neural Network by Tariq Rashid.
    bias_layer = numpy.zeros((layerb, 1))
    weights.append(weight_layer)
    bias.append(bias_layer)
##################################################

plot_x = []
plot_y = []
for itno in range(no_of_iters): 
            i=0
            while(i<len(yTrain)):
                x_batch = xTrain[i:i+size_of_batch]
                y_batch = yTrain[i:i+size_of_batch]
                i = i+size_of_batch
                plain_values, activated_values = NN.forwardpropogation(x_batch, weights,activations,bias)
                if itno%1000 == 0 or itno==0:
                    plot_x.append(itno)
                    loss = NN.loss_function(activated_values[-1],y_batch)
                    plot_y.append(loss)
                    print('loss = '+str(loss))
                delta = NN.backpropagation(y_batch, plain_values, activated_values, weights, activations,no_of_layers)
                for layer in range(no_of_layers-1):
                    weights[layer] = weights[layer] - learning_rate * delta["dweights"+str(layer)]
                    bias[layer] = bias[layer] - learning_rate * delta["dbias"+str(layer)]            
                
NN.evaluate(xTrain, yTrain, weights, activations, bias, "training6")
NN.evaluate(xTest, yTest, weights, activations, bias,"testing6")
if plot:
        plt.plot(plot_x,plot_y)
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.title("Loss Function convergence")
        plt.savefig("Image/lossfunction.png")






