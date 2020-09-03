from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import math


iris = load_iris()
iris_x = iris.data
iris_y = iris.target

train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size = 0.3)


class BernoulliNB2:
    """每個feature設定一個threshold以list傳入"""
    def __init__(self, features, targets, threshold):
        self.features = features
        self.targets = targets
        self.threshold = threshold
        self.total = len(features)
    
    """因為使用伯努力Naive Bayes，先將資料二值化"""
    def clean(self):
        for i in range(len(self.threshold)):
            t = self.threshold[i]
            for j in range(self.total):
                if self.features[j][i]>=t:
                    self.features[j][i] = 1
                else:
                    self.features[j][i] = 0

    
    def predict(self, classes, test):
        #紀錄feature label為1的個數
        self.px = np.zeros(len(self.features[0]))
        for i in self.features:
            for j in range(len(i)):
                if i[j] == 1:
                    self.px[j]+=1
        
        #讓test data的feature根據門檻值轉為0/1兩類
        for i in range(len(self.threshold)):
            t = self.threshold[i]
            for j in range(len(test)):
                if test[j][i]>=t:
                    test[j][i] = 1
                else:
                    test[j][i] = 0
        ans = np.zeros(len(test))
        for n, data in enumerate(test,0):
            prob = np.zeros(len(classes))
            #計算第j個feature為k的條件下，類別為i的機率
            for i in range(len(prob)):
                for j in range(len(data)):
                    if data[j] == 1:
                        px = 0
                        for x,y in enumerate(self.features,0):
                            if y[j] == 1 and self.targets[x]==i:
                                px +=1
                        prob[i] += math.log((px/self.px[j])+0.001)
                    else:
                        px = 0
                        for x,y in enumerate(self.features,0):
                            if y[j] == 0 and self.targets[x]==i:
                                px +=1
                        prob[i] += math.log((px/(self.total-self.px[j]))+0.001)
            #print(prob)
            ans[n] = np.argmax(prob)
        return ans
print(" ")
#call sklearn
tStart = time.time()
model = BernoulliNB(binarize=[5.8,3,4.35,1.3])
model.fit(train_x,train_y)
print("sklearn Acc: %0.3f"%(model.score(test_x, test_y)))
print("sklearn: %0.3f sec"%(time.time()-tStart))


tStart = time.time()
model = BernoulliNB2(train_x, train_y, [5.8,3,4.35,1.3])
model.clean()
acc = 0
ans = model.predict([0,1,2],test_x)
for i, data in enumerate(ans):
    if data == test_y[i]:
        acc += 1
print("HandCut Acc: %0.3f"%(acc/len(ans)))
print("HandCut: %0.3f sec"%(time.time()-tStart))

print(" ")