#!/opt/share/bin/anaconda3/bin python3
import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use("Agg")
import xgboost as xgb
import shap

dataset_train=np.load("./trainset.npz")
X1,Y1=dataset_train["data"],dataset_train["label"]
# X1,X2,Y1,Y2= train_test_split(X1,Y1,test_size=0.1)
dataset_test=np.load("./testset.npz")
X2,Y2=dataset_test["data"],dataset_test["label"]

scaler = preprocessing.MinMaxScaler().fit(X1)
trainX = scaler.transform(X1)
# scaler = preprocessing.MinMaxScaler().fit(X2)
# testT = scaler.transform(X2)
X_train = np.array(trainX[:100])
y_train = np.array(Y1[:100])
# testdata = np.array(X2)
# testlabel = np.array(Y2)

# X_train, X_test, y_train, y_test = train_test_split(traindata, trainlabel,stratify=trainlabel, test_size=0.2, random_state=42)
xlf = xgb.XGBClassifier(max_depth=4,
                        learning_rate=0.02,
                        n_estimators=300,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=10,
                        max_delta_step=1,
                        subsample=0.6,
                        colsample_bytree=0.9,
                        colsample_bylevel=1,
                        reg_alpha=0.25,
                        reg_lambda=0,
                        scale_pos_weight=1,
                        seed=42,
                       missing=None)

xlf.fit(X_train, y_train)
#plot shap feature importance
explainer =shap.TreeExplainer(xlf)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
# shap.summary_plot(shap_values, X_train, plot_type="bar")
# xlf_shap_import = np.mean(abs(shap_values),axis=0) #this is the shap importance for each feature based on all train data
# #sort the shap importance
# idx_sorted = np.argsort(xlf_shap_import) #this is ascend
# np.save('sorted.npz',idx_sorted)