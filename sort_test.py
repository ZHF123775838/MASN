import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import xgboost
matplotlib.use("Agg")
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)


dataset_train=np.load("./trainset.npz")
X1,Y1=dataset_train["data"],dataset_train["label"]
# X1,X2,Y1,Y2= train_test_split(X1,Y1,test_size=0.1)
dataset_test=np.load("./testset.npz")
X2,Y2=dataset_test["data"],dataset_test["label"]

scaler = preprocessing.MinMaxScaler().fit(X1)
trainX = scaler.transform(X1)
scaler = preprocessing.MinMaxScaler().fit(X2)
testT = scaler.transform(X2)
X_train = np.array(trainX)
y_train = np.array(Y1)
X_test = np.array(testT)
y_test = np.array(Y2)

idx_sorted=np.load('sorted.npz.npy')
result_list=[]
for i in range(5,len(idx_sorted),5):
    X_train_tmp = X_train[:,idx_sorted[i:]]
    X_test_tmp = X_test[:,idx_sorted[i:]]
    rlf = xgboost.XGBClassifier() #RandomForestClassifier(random_state=42)
    # rlf  = GridSearchCV(rfc_tmp,param_grid=param_grid, scoring=scorings, cv=5, refit='f1',verbose=1, return_train_score=True,n_jobs=-1)
    rlf.fit(X_train_tmp,y_train)
    expected = y_test
    predicted = rlf.predict(X_test_tmp)
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average="binary") 
    precision = precision_score(expected, predicted , average="binary") #精确率
    f1 = f1_score(expected, predicted , average="binary") 
    tn, fp, fn, tp= metrics.confusion_matrix(expected, predicted).ravel()
    # print(cm,cm[0][0],cm[0][1])  #混淆矩阵
    result=np.array([i,'%.4f'%accuracy,'%.4f'%recall,'%.4f'%precision,'%.4f'%f1,tn,fp,fn,tp])
    result_list.append(result)
np.save('result_xgb.npy',np.array(result_list))