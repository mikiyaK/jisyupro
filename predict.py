#ffnn.py
from __future__ import print_function #for using python3's print_function in python2
from layer import * #import all members from layer.py
import util #import util.py
import time
import random
import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import math
import csv
from mpl_toolkits.mplot3d import Axes3D
xmax = 1000
vmax = 10
in_sequence_len = 3
out_sequence_len = 3
input_dim = 2
output_dim = 2
a = 220
b = 810
c = 410
def read_data(path, in_sequence_len, out_sequence_len):
    data, target = [], []
    filenames = os.listdir(path)
    for filename in filenames:
        x = pd.read_csv(path + '/' + filename, header=None)
        v = pd.DataFrame(index=np.arange(len(x)-1),columns=[0, 1, 2],dtype=float)
        for i in range(len(x)):
            x.iloc[i,2] = 480 - x.iloc[i,2]
            x.iloc[i,1] = (b * c) * (x.iloc[i,1] - 320) / ((a - b) * x.iloc[i,2] + b * c)
            x.iloc[i,2] = x.iloc[i,2] * 215 / 350
            #dt = x.iloc[i+1,0] - x.iloc[i,0]
            #v.iloc[i,1:3] = (x.iloc[i+1,1:3] - x.iloc[i,1:3]) / dt
            #v.iloc[i,0] = x.iloc[i,0]
        print(x)  
        x = x / xmax
        v = v / vmax
        #print(x)
        #print(v)
        for i in range(len(x) - in_sequence_len):
            tmp = []
            for j in range(in_sequence_len):
                tmp.append(x.iloc[i+j,1])
                tmp.append(x.iloc[i+j,2])
               # tmp.append(v.iloc[i+j,1])
               # tmp.append(v.iloc[i+j,2])
            data.append(tmp)
            tmp = []
            #for j in range(in_sequence_len, in_sequence_len+out_sequence_len):
            tmp.append(x.iloc[i+in_sequence_len,1])
            tmp.append(x.iloc[i+in_sequence_len,2])
               # tmp.append(v.iloc[i+j,1])
               # tmp.append(v.iloc[i+j,2])
            target.append(tmp)
    re_data = np.array(data)
    re_target = np.array(target)
    return re_data, re_target


def predict(model, path, filename):
    x = pd.read_csv(path + '/' + filename, header=None)
    steps = len(x) - in_sequence_len
    v = pd.DataFrame(index=np.arange(len(x)-1),columns=[0, 1, 2],dtype=float)
    for i in range(len(x)):
        x.iloc[i,2] = 480 - x.iloc[i,2]
        x.iloc[i,1] = (b * c) * (x.iloc[i,1] - 320) / ((a - b) * x.iloc[i,2] + b * c)
        x.iloc[i,2] = x.iloc[i,2] * 215 / 350
    x = x / xmax
    v = v / vmax
    data = []
    tmp = []
    for j in range(in_sequence_len):
        tmp.append(x.iloc[j,1])
        tmp.append(x.iloc[j,2])
    data.append(tmp)
    data = np.array(data)
    future_result = data
    for i in range(steps):
        predict = model.forward(data)
        #print(predict)
        data = np.delete(data, list(range(output_dim)))
        data = np.append(data, predict).reshape(1, in_sequence_len*input_dim)
        future_result = np.append(future_result, predict)
    future_result = future_result.reshape(steps+in_sequence_len, output_dim)
    predicted = pd.DataFrame(data=future_result, columns=['x','y'],dtype='float')

    for i in range(in_sequence_len):
        predicted.loc[i,'x'] = x.iloc[i,1]
        predicted.loc[i,'y'] = x.iloc[i,2]
    predicted['x'] *= xmax
    predicted['y'] *= xmax
    predicted.to_csv('result'+'/'+'csv'+'/'+filename, index = False)
    #predicted.to_csv('result'+'/'+'csv_linear'+'/'+filename, index = False)
def predict_all(model, path):
    filenames = os.listdir(path)
    for filename in filenames:
        predict(model, path, filename)

def draw_graph(true_path, result_path):
    filenames = os.listdir(true_path)
    error = []
    for filename in filenames:
        true = pd.read_csv(true_path + '/' + filename,header=None,names=['t','x','y'])
        v = pd.DataFrame(index=np.arange(len(true)-1),columns=['t', 'x', 'y'])
        for i in range(len(true)):
            true.loc[i,'y'] = 480 - true.loc[i,'y']
            true.loc[i,'x'] = (b * c) * (true.loc[i,'x'] - 320) / ((a - b) * true.loc[i,'y'] + b * c)
            true.loc[i,'y'] = true.loc[i,'y'] * 215 / 350
            #dt = true.loc[i+1,'t'] - true.loc[i,'t']
            #v.loc[i,['x','y']] = (true.loc[i+1,['x','y']] -true.loc[i,['x','y']]) / dt
            #v.loc[i,'t'] = true.loc[i,'t']
        predicted = pd.read_csv(result_path + '/' + 'csv' + '/' + filename)
        #predicted = pd.read_csv(result_path + '/' + 'csv_linear' + '/' + filename)
        #print(predicted['x'])
        #print(predicted['y'])
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Prediction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.scatter(true['x'],true['y'],c='blue', label='truth', alpha=0.3, s=10)
        plt.scatter(predicted['x'],predicted['y'],c='red', label='predicted', alpha=0.3, s=10)
        ax.legend(loc='lower left')
        savename = 'result/graph/' + filename.split('.csv')[0] + '.png'
        #savename = 'result/graph_linear/' + filename.split('.csv')[0] + '.png'
        plt.savefig(savename)
        plt.close()
        '''error.append(np.sqrt((true.iloc[len(true)-1,1]-predicted.iloc[len(true)-1,1])**2
                             +(true.loc[len(true)-1,2]-predicted.iloc[len(true)-1,2])**2
        ))'''
    '''Ave = []
    Ave.append(np.mean(error))
    plt.figure()
    ax.set_title('The error of prediction')
    ax.set_ylabel('distance')
    label = ["predicted"]
    plt.bar([1], Ave, yerr=error, ecolor="black", tick_label=label, align="center")
    plt.savefig('result/evaluation/evaluation.png')'''
    
def main(epoch):
    model = Sequential()   #model composes the sequential of hidden layers of the neural network by using Sequential() which is a class of layer.py
    model.addlayer(FlattenLayer())
    model.addlayer(LinearLayer(input_dim*in_sequence_len, 24))
    #model.addlayer(ReLULayer())
    model.addlayer(LinearLayer(24, 24))
    #model.addlayer(ReLULayer())
    model.addlayer(LinearLayer(24, output_dim))

    classifier = Classifier(model) #to make model work as a system by using Classifier() which is a class of layer.py

    data_train, target_train = read_data("data_linear", in_sequence_len, out_sequence_len)
    data_test = data_train
    target_test = target_train
    batchsize = 50
    print(len(data_train))
    ntrain = len(data_train)
    ntest = len(data_test)

    for e in range(epoch):
        print('epoch %d'%e)
        randinds = np.random.permutation(ntrain) #create an array which includes numbers 0 to ntrain randomly
        for i_train in range(0, ntrain, batchsize):
            ind = randinds[i_train:i_train+batchsize]
            x = data_train[ind]
            t = target_train[ind]
            start = time.time()
            loss = classifier.update(x, t)
            end = time.time()
            print('train iteration %d, elapsed time %f, loss %f, acc not-define'%(i_train//batchsize, end-start, loss))

    start = time.time()
    acctest = 0
    losstest = 0
    for i_test in range(0, ntest, batchsize):
            x = data_test[i_test:i_test+batchsize]
            t = target_test[i_test:i_test+batchsize]
            loss = classifier.predict(x,t)
            losstest += loss
    losstest /= (ntest // batchsize)
    end = time.time()
    print('test, elapsed time %f, loss %f, acc not-define'%(end-start, loss))
    #print(model.layers[1].params['W'])
    #print(model.layers[1].params['b'])
    '''with open('LinearLayer_1_W.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(model.layers[1].params['W'])
    with open('LinearLayer_1_b.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(model.layers[1].params['b'])
    with open('LinearLayer_2_W.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(model.layers[2].params['W'])
    with open('LinearLayer_2_b.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(model.layers[2].params['b'])
    with open('LinearLayer_3_W.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(model.layers[3].params['W'])
    with open('LinearLayer_3_b.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(model.layers[3].params['b'])'''
    predict_all(model, "data")
    #predict_all(model, "data_linear")

    draw_graph("data", "result")
    #draw_graph("data_linear", "result")
    
if __name__ == '__main__': 
    epoch = 1000000
    main(epoch)
 

        
