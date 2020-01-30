import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import csv

class k_means():

    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.rep_point = np.zeros((k,data.shape[1]))
        self.result = np.zeros(data.shape[0])
        random = np.random.randint(0,data.shape[0]-1,k)
        for i in range(k):
            self.rep_point[i] = data[random[i]]

    def clustering(self):
        delta = 1.0
        while delta > 0.000001:
            dist = np.sum(self.rep_point ** 2, axis=1, keepdims=True).T - 2 * np.dot(self.data, self.rep_point.T)
            self.result = np.argsort(dist)[:,0].T
            old_rep_point = self.rep_point.copy()
            for i in range(self.k):
                int = np.where(self.result == i)
                tem = self.data[int]
                self.rep_point[i] = np.mean(tem, axis=0)
            delta = np.sum(np.sum((self.rep_point - old_rep_point) ** 2, axis=1))
            print(delta)
        return self.result, self.rep_point

def main():
   # with open('calib_dataset.csv') as f:
    #    reader = csv.reader(f)
     #   data = [row for row in reader]
    data = np.loadtxt('calib_dataset.csv', delimiter=',')
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for k in range(38,40):
        np.random.seed(0)
        clustering_machine = k_means(k,data)
        [result,rep_point] = clustering_machine.clustering()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('k-means(k={})'.format(k))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        for i in range(k):
            int = np.where(result==i)
            tem = data[int].T
            plt.scatter(rep_point[i][0],rep_point[i][1],c="r",s=300,marker="*")
            plt.scatter(tem[0],tem[1],c="b", label='label{}'.format(i))
        savename = 'graph/k-means(k={})'.format(k)
        plt.savefig(savename)
        plt.close()
          


if __name__ == '__main__':
    main()
                    
                    
                
                
