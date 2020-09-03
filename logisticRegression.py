from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import math


iris = load_iris()
iris_x = iris.data
iris_y = iris.target


train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size = 0.3)
y_class1 = np.zeros(len(train_y))
y_class2 = np.zeros(len(train_y))
y_class3 = np.zeros(len(train_y))
for i, data in enumerate(train_y):
    y_class1[i]=1 if data == 0 else 0
    y_class2[i]=1 if data == 1 else 0
    y_class3[i]=1 if data == 2 else 0
#print(y_class1, y_class2, y_class3, train_y)

class logisticRegression:
    def __init__(self, features):
        self.features = features
        self.input = np.ones((1, features+1))
        self.params = np.random.rand(features+1, 1)
    
    def train(self, epoch, data, target):
        for k in range(epoch):
            self.loss = 0
            for m, n in enumerate(data):
                x, y = n, target[m]
                self.input[0][1:] = x
                self.yhat = np.dot(self.input, self.params)
                self.yhat = 1/(1+np.exp(self.yhat))
                self.target = np.zeros(self.yhat.shape)
                self.target[0][0] = y
                self.loss += -np.dot(self.target, np.log(self.yhat))-np.dot(np.ones(self.target.shape)-self.target,np.log(np.ones(self.yhat.shape)-self.yhat))
                for i, w in enumerate(self.params):
                    self.params[i][0] = w - 0.01*(y-self.yhat[0][0])*self.input[0][i]
            #print("epoch: ",k, "loss: ",self.loss)
    
    def test(self, inputs):
        ans = np.zeros(len(inputs))
        for i, data in enumerate(inputs):
            self.input[0][1:] = data
            self.yhat = np.dot(self.input, self.params)
            self.yhat = 1/(1+np.exp(self.yhat))
            ans[i] = self.yhat[0][0]
        return ans

tStart = time.time()
model1 = logisticRegression(4)
model1.train(100, train_x, y_class1)
p1 = model1.test(test_x)
model2 = logisticRegression(4)
model2.train(100, train_x, y_class2)
p2 = model2.test(test_x)
model3 = logisticRegression(4)
model3.train(100, train_x, y_class3)
p3 = model3.test(test_x)
ans = []
for n in range(len(p1)):
    if p1[n]>=p2[n] and p2[n]>=p3[n]:
        ans.append(0)
    elif p2[n]>=p1[n] and p2[n]>=p3[n]:
        ans.append(1)
    else:
        ans.append(2)

hit = 0
for i, data in enumerate(ans,0):
    hit+=1 if data == test_y[i] else 0
accHandcut = "handcut accuracy:%0.3f "%(hit/len(ans))
tHandcut = "handcut:%0.3f sec"%(time.time()-tStart)
print(" ")
print(tHandcut)
print(accHandcut)


#sklearn
tStart = time.time()
model_1 = LogisticRegression()
model_1.fit(train_x, y_class1)
model_2 = LogisticRegression()
model_2.fit(train_x, y_class2)
model_3 = LogisticRegression()
model_3.fit(train_x, y_class3)
p1 = model_1.predict_proba(test_x)
p2 = model_2.predict_proba(test_x)
p3 = model_3.predict_proba(test_x)
ans = []
for n in range(len(p1)):
    if p1[n][1]>=p2[n][1] and p2[n][1]>=p3[n][1]:
        ans.append(0)
    elif p2[n][1]>=p1[n][1] and p2[n][1]>=p3[n][1]:
        ans.append(1)
    else:
        ans.append(2)
hit = 0
for i, data in enumerate(ans,0):
    hit+=1 if data == test_y[i] else 0
accSklearn = "sklearn acc:%0.3f "%(hit/len(ans))
tSklearn = "sklearn:%0.3f sec"%(time.time()-tStart)

print("+"*20)
print(tSklearn)
print(accSklearn)
print(" ")