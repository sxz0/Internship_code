import os
import statistics
import time

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
#import umap

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#General parameters for the experiments
n_devices=32
dataset_dir="../datasets"
n_samples_device = 200 #10000
window=100
test_size=0.5

#Dataset to be read and processed
dataset_name="second_collection.csv"


df=pd.read_csv(dataset_dir+"/"+dataset_name, index_col=False, header=None)
final_df = pd.DataFrame()
df_X = df.iloc[:, :-1]
df_Y = df.iloc[:, -1:]
print(df.describe())
df_X = df_X.iloc[:, [1]]  # 0,1,2

for n in range(0,df.shape[0]//n_samples_device):

    temp_df = pd.DataFrame()

    df_X_selec = df_X[n * n_samples_device:n * n_samples_device + n_samples_device]
    df_Y_selec = df_Y[n * n_samples_device:n * n_samples_device + n_samples_device]

    temp_df = temp_df.append(df_Y_selec)

    temp_df["mean"] = df_X_selec.rolling(window).mean()
    temp_df["min"]=df_X_selec.rolling(window).min()
    temp_df["max"]=df_X_selec.rolling(window).max()
    temp_df["median"] = df_X_selec.rolling(window).median()
#    temp_df["stdev"] = df_X_selec.rolling(window).std()
#    temp_df["skew"] = df_X_selec.rolling(window).skew()
#    temp_df["kurt"] = df_X_selec.rolling(window).kurt()
    temp_df["sum"] = df_X_selec.rolling(window).sum()

    temp_df["Y"] = df_Y_selec

    temp_df = temp_df.iloc[:, 1:]

    final_df = final_df.append(temp_df)

print(final_df.shape)
final_df.dropna(inplace=True)

print(final_df)
print(final_df.shape)

df_X=final_df.iloc[:,:-1]
df_Y=final_df.iloc[:,-1:]
X_train,X_test, y_train,y_test = train_test_split(df_X,df_Y, test_size=test_size,shuffle=False)#, stratify=df_Y) #IMPORTANT

#rf = RandomForestClassifier(n_estimators = 100)
#rf = DecisionTreeClassifier()
rf = XGBClassifier(min_child_weight= 5, max_depth= 20, learning_rate= 0.1, gamma= 0.01, colsample_bytree= 0.5)
#rf = KNeighborsClassifier(n_neighbors=8)
#rf = GaussianNB()
#rf= svm.SVC(kernel='rbf')
rf.fit(X_train, y_train)

pred=rf.predict(X_test)
accuracy=rf.score(X_test,y_test)
print("Accuracy: {}".format(accuracy))
print(classification_report(y_test, pred, target_names=rf.classes_))

array=confusion_matrix(y_test, pred)
df_cm = pd.DataFrame(array)
plt.figure(figsize = (35,35))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g',xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.show()

feat_labels=X_train.columns
importances=rf.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))