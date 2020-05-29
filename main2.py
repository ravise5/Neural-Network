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
no_of_layers = 4
no_of_nodes = [10, 8, 8,1]
data_scaler = 'min-max'
activations = ['relu','relu','sigmoid']
no_of_iters = 100000
size_of_batch = 1168
learning_rate = 0.00001
dropout_rate = 0.0
plot = False
#####################################################

if data_scaler == 'standardization':
    x = NN.standardization(x)
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
    weight_layer = numpy.random.randn(layerb,layera)/numpy.sqrt(layera)
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
                dropout_weights, dropout_bias, x_batch, active_nodes = NN.dropout(dropout_rate, weights, x_batch, bias)
                plain_values, activated_values = NN.forwardpropogation(x_batch, dropout_weights,activations,dropout_bias)
                if itno%1000 == 0 or itno==0:
                    plot_x.append(itno)
                    loss = NN.loss_function(activated_values[-1],y_batch)
                    plot_y.append(loss)
                    print('loss = '+str(loss))
                delta = NN.backpropagation(y_batch, plain_values, activated_values, dropout_weights, activations,no_of_layers)
                for layer in range(no_of_layers-1):
                    dropout_weights[layer] = dropout_weights[layer] - learning_rate * delta["dweights"+str(layer)]
                    dropout_bias[layer] = dropout_bias[layer] - learning_rate * delta["dbias"+str(layer)]
                weights, bias = NN.updateWeights(weights, dropout_weights, active_nodes, bias, dropout_bias)
                
NN.evaluate(xTrain, yTrain, weights, activations, bias, "training5")
NN.evaluate(xTest, yTest, weights, activations, bias,"testing5")
if plot:
    plt.plot(plot_x,plot_y)
    plt.savefig("Image/lossfunction.png")






