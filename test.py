import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)


dataset_train=np.load("./trainset.npz")
X1,Y1=dataset_train["data"],dataset_train["label"]
# X1,X2,Y1,Y2= train_test_split(X1,Y1,test_size=0.1)
dataset_test=np.load("./testset.npz")
X2,Y2=dataset_test["data"],dataset_test["label"]

scaler = MinMaxScaler().fit(X1)
traindata = scaler.transform(X1)
scaler = MinMaxScaler().fit(X2)
testdata = scaler.transform(X2)
trainlabel = np.array(Y1)
testlabel = np.array(Y2)

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# X = traindata
y = trainlabel

X_train = np.array(traindata)
y_train = np.array(Y1)
X_test = np.array(testdata)
y_test = np.array(Y2)

idx_sorted=np.load('sorted.npz.npy')
i=410
X = X_train[:,idx_sorted[i:]]
X_test_tmp = X_test[:,idx_sorted[i:]]
rlf = RandomForestClassifier(random_state=42)

# 训练随机深林模型
# rf = RandomForestClassifier(random_state=42)
rf= XGBClassifier()
rf.fit(X, y)
expected = testlabel
predicted = rf.predict(X_test_tmp)
print(predicted)
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary") 
precision = precision_score(expected, predicted , average="binary") #精确率
f1 = f1_score(expected, predicted , average="binary") 
cm = metrics.confusion_matrix(expected, predicted)
# print(cm,cm[0][0],cm[0][1])  #混淆矩阵
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("tpr","%.3f" %tpr)
print("fpr","%.3f" %fpr)
print("Accuracy","%.3f" %accuracy)
print("precision","%.3f" %precision)
print("recall","%.3f" %recall)
print("f-score","%.3f" %f1)
print("fpr","%.3f" %fpr)
print("tpr","%.3f" %tpr)