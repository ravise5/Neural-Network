import numpy
import math
import pandas as pd
import random 

def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))

def linear(inpt):
    return inpt

def tanh(inpt):
    return numpy.tanh(inpt)

def relu(inpt):
    result = inpt
    result[inpt<0] = 0
    return result

def derivative_sigmoid(inpt):
    sig = sigmoid(inpt)
    return sig * (1 - sig)

def derivative_linear(inpt):
    return 1

def derivative_tanh(inpt):
    t = tanh(inpt)
    return (1 - (t**2))

def derivative_relu(inpt):
    inp = numpy.copy(inpt)
    inp[inp<=0] = 0
    inp[inp>0] = 1
    return inp

def forwardpropogation(x, weights, activations, bias):
        r1 = numpy.copy(x)
        r1 = r1.T
        plain_values = []
        activated_values = [r1]
        for i in range(len(weights)):
            r1 = numpy.matmul(weights[i], r1) + bias[i]
            plain_values.append(r1)
            if activations[i] == "relu":
                r1 = relu(r1)
            elif activations[i] == "sigmoid":
                r1 = sigmoid(r1)
            elif activations[i] == "linear":
                r1 = linear(r1)
            elif activations[i] == "tanh":
                r1 = tanh(r1)
            activated_values.append(r1)
        return (plain_values, activated_values)

def backpropagation(y, plain_values, activated_values, weights, activations, layers):
        delta = {}
        batch_size = y.shape[1]
        A = activated_values[-1]
        L = layers
        y=y.T
        dA = -numpy.divide(y,A) + numpy.divide(1-y,1-A)
        dZ = dA * derivative_sigmoid(plain_values[L-2])
        dW = dZ.dot(activated_values[L-2].T) / batch_size
        db = numpy.sum(dZ, axis=1, keepdims=True) / batch_size
        dAPrev = (weights[L-2].T).dot(dZ)
        
        delta["dweights" + str(L-2)] = dW
        delta["dbias" + str(L-2)] = db

        for l in range(layers - 1, 1, -1):
            dZ = dAPrev * derivative_sigmoid(plain_values[l-2])
            dW = 1. / batch_size * dZ.dot(activated_values[l-2].T)
            db = 1. / batch_size * numpy.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = weights[l-2].T.dot(dZ)

            delta["dweights" + str(l-2)] = dW
            delta["dbias" + str(l-2)] = db

        return delta        
    
def loss_function(prediction, actual):
   actual=actual.T
   size = actual.shape[1]
   loss = numpy.multiply(numpy.log(prediction),actual) + numpy.multiply(1 - actual, numpy.log(1 - prediction))
   loss = - numpy.sum(loss)
   return  loss / size

def dropout(rate, weights, x, bias):
    drop_weights=[]
    drop_bias = []
    active_nodes = []
    rate = 1 - rate         #?
    input_nodes = weights[0].shape[1]
    active_input_nodes = int(input_nodes * 1)
    active_prev_layer = sorted(random.sample(range(0, input_nodes), 
                              active_input_nodes))
    active_nodes.append(active_prev_layer)
    x = x[:,active_prev_layer]
    for i in range(len(weights)-1):
        w = weights[i]
        nodes = weights[i].shape[0]
        active_layer_nodes = int(nodes * rate)
        active_new_layer = sorted(random.sample(range(0, nodes), 
                              active_layer_nodes))
        w = w[:,active_prev_layer]
        w = w[active_new_layer]
        drop_bias.append(bias[i][active_new_layer])
        drop_weights.append(w)
        active_prev_layer = active_new_layer
        active_nodes.append(active_prev_layer)
    
    w = weights[-1]
    w = w[:,active_prev_layer]
    drop_bias.append(bias[-1])
    drop_weights.append(w)
    
    return drop_weights, drop_bias, x, active_nodes

def updateWeights(weights, dropout_weights, active_nodes, bias, dropout_bias):
    
    for i in range(len(weights)-1):
        temp = weights[i][:,active_nodes[i]]
        temp[active_nodes[i+1]] = dropout_weights[i]
        weights[i][:,active_nodes[i]] = temp
        bias[i][active_nodes[i+1]] = dropout_bias[i]
        
    weights[-1][:,active_nodes[-1]] = dropout_weights[-1]
    
    return weights, bias
        
def evaluate(x, y, weights, activations, bias, name):
    plain_values, activated_values = forwardpropogation(x, weights,activations,bias)
    predictions = activated_values[-1]
    pred = numpy.copy(predictions)
    ylabel = []
    for i in range(pred.shape[1]):
        if pred[0][i] > 0.5:
            ylabel.append(1)
        else:
            ylabel.append(0)

    ytest = y
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    metrics = pd.Series([])
    value = pd.Series([])
    
    for i in range(len(ylabel)):
        if ytest[i] == ylabel[i] and ytest[i] == 1 :
            tp = tp+1
        elif ytest[i] == ylabel[i] and ytest[i] == 0:
            tn = tn+1
        if ytest[i] != ylabel[i] and ytest[i] == 1 :
            fn = fn + 1
        elif ytest[i] != ylabel[i] and ytest[i] == 0:
            fp = fp + 1

    print("True Positives = "+str(tp))
    print("False Positives = "+str(fp))
    print("True Negatives = "+str(tn))
    print("False Negatives = "+str(fn))

    Precision = tp/(tp+fp)
    Accuracy = (tp+tn)/(tp+tn+fp+fn)
    Recall = tp/(tp+fn)
    
    F1_score = 2*Precision*Recall/(Precision+Recall)
    FPR = fp/(fp+tn)
    Specificity = 1 - FPR
    print("Precision = "+str(Precision))
    print("Accuracy = "+str(Accuracy))
    print("Recall = "+str(Recall))

    print("F1 Score = "+str(F1_score))
    print("FPR = "+str(FPR))
    print("Specificity = "+str(Specificity))
    
    metrics[0] = "True Positives"
    metrics[1] = "False Positives"
    metrics[2] = "True Negatives"
    metrics[3] = "False Negatives"
    metrics[4] = "Precision"
    metrics[5] = "Accuracy"
    metrics[6] = "Recall"
    metrics[7] = "F1 Score"
    metrics[8] = "FPR"
    metrics[9] = "Specificity"

    value[0] = tp
    value[0] = fp
    value[0] = tn
    value[0] = fn
    value[0] = Precision
    value[0] = Accuracy
    value[0] = Recall
    value[0] = F1_score
    value[0] = FPR
    value[0] = Specificity
    
    resdf = pd.DataFrame({
        "TP":tp,
        "FP":fp,
        "TN":tn,
        "FN":fn,
        "Precision":Precision,
        "Accuracy":Accuracy,
        "Recall":Recall,
        "F1_score":F1_score,
        "FPR":FPR,
        "Specificity":Specificity
        },
        index=[0]
    )

    #resdf.to_csv("Result/Dropout/"+name+"_metrics.csv")

def minmax(data):
        data = data.astype(float)
        maxElement = numpy.amax(data, axis=0)
        minElement = numpy.amin(data, axis=0)        
        data = (data - minElement)/(maxElement-minElement)
    
        return data

def standardization(data):
        data = data.astype(float)
        mean = numpy.mean(data, axis=0)
        std = numpy.std(data, axis=0)        
        data = (data - mean)/(std)
    
        return data

    



