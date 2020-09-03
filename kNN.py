from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import math
import numpy as np
# loading iris data
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# split training and testing data
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)
train_x = train_X
test_x = test_X

#sklearn
tStart = time.time()
plts = []
for i in range(100):
    clf = neighbors.KNeighborsClassifier(n_neighbors=i+1)
    iris_clf = clf.fit(train_X, train_y)

    test_y_predicted = iris_clf.predict(test_X)
    error = 0
    for j in range(len(test_y)):
        if test_y[j]!=test_y_predicted[j]:
            error +=1
    acc = (len(test_y)-error) / len(test_y)
    plts.append(acc)


tSklearn = "sklearn:%0.3f sec"%(time.time()-tStart)


class KNN:
    def __init__(self, numberOfk):
        self.k = numberOfk
    
    """training process of knn just store the training data"""
    def Train(self, data_x, data_y):
        self.data_x = data_x
        """normalize"""
        for i in range(len(self.data_x[0])):
            self.min = min(self.data_x[:][0])
            self.diff = max(self.data_x[:][0]) - self.min
            if self.diff == 0:
                continue #divided by zero
            self.data_x[:][0] = 100*(self.data_x[:][0]-self.min)/self.diff
        self.data_y = data_y
    
    """predict the class of each test data and return the answer"""
    def Test(self, data_x):
        self.test = data_x
        dtype = [('distance',float),('class', int)]
        self.ans = []
        for x in data_x:
            distance = []
            for item in range(len(self.data_x)):
                _distance = 0
                for i in range(len(x)):
                    _distance += (x[i]-self.data_x[item][i])**2
                dcset = (math.sqrt(_distance),self.data_y[item])
                distance.append(dcset)
            distance = np.array(distance, dtype = dtype)
            distance = np.sort(distance, order='distance')
            #print(distance)
            _ans = []
            for j in range(0,self.k):
                _ans.append(distance[j][1])
            _ans = max(_ans, key = _ans.count)
            self.ans.append(_ans)
        return self.ans


tStart = time.time()
plts1 = []
for i in range(100):
    myKNN = KNN(i+1)
    myKNN.Train(train_x, train_y)
    predict_y = np.array(myKNN.Test(test_x))
    error = 0
    for j in range(len(test_y)):
        if test_y[j]!=predict_y[j]:
            error +=1
    acc = (len(test_y)-error) / len(test_y)
    str1 = "%3d %0.3f"%(i,acc)
    #print(str1)
    plts1.append(acc)
tKnn = "handcut:%0.3f sec"%(time.time()-tStart)


plt.figure(0)
plt.plot(range(len(plts)), plts ,label='acc')
plt.title('KNN(handcut)')

lines = plt.plot(range(len(plts)), plts, range(len(plts1)), plts1)  

plt.legend((lines[0],lines[1]),(tSklearn, tKnn), loc='upper right')  
plt.title('kNN')  


plt.savefig('kNN_Iris.png',dpi=300,format='png')
plt.close()
