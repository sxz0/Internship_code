import os
import statistics
import time

#from combo.utils.utility import standardizer
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
#from pyod.models.hbos import HBOS
#from pyod.models.cblof import CBLOF
#from pyod.models.loci import LOCI
#from pyod.models.xgbod import XGBOD
#from pyod.models.cof import COF
#from pyod.models.loda import LODA
#from pyod.models.copod import COPOD
#from pyod.models.sod import SOD
#from pyod.models.vae import VAE
#from pyod.models.lof import LOF,LocalOutlierFactor
#from pyod.models.lscp import LSCP
##from pyod.models.so_gaal import SO_GAAL
#from pyod.models.mo_gaal import MO_GAAL
#from pyod.models.iforest import IsolationForest,IForest
#from pyod.models.ocsvm import OCSVM
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.knn import KNN
#from pyod.models.combination import moa,aom,median,majority_vote
#from pyod.models.abod import ABOD
import seaborn as sb
from scipy.stats import gaussian_kde

import warnings
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

#General parameters for the experiments
n_devices=32
dataset_dir="../datasets"
n_samples_device = 200 #10000
window=50
n_samples_device_window=n_samples_device-window+1
test_size=0.2
contamination=0.2

#Dataset to be read and processed
dataset_name="current_dataset.csv"
mac_model_file="../MAC-Model.txt"

df=pd.read_csv(dataset_dir+"/"+dataset_name, index_col=False, header=None)
final_df = pd.DataFrame()
df_X = df.iloc[:, :-1]
df_Y = df.iloc[:, -1:]
print(df.describe())
df_X = df_X.iloc[:, [1]]  # 0,1,2

#### Add model to the label to add clarity in the plots #####
mac_model={}
with open(mac_model_file) as f:
    for line in f:
        p=line.split(" ")
        mac_model[p[0]]=p[3]
df_Y[7]=df_Y[7].apply(lambda x: mac_model[str(x)]+"_"+str(x))

list_TP = []
list_FP = []
# list_fing=[]
list_train = []
list_test = []
labels = []

for n in range(0,df.shape[0]//n_samples_device):

    temp_df = pd.DataFrame()

    df_X_selec = df_X[n * n_samples_device:n * n_samples_device + n_samples_device]
    df_Y_selec = df_Y[n * n_samples_device:n * n_samples_device + n_samples_device]

    temp_df = temp_df.append(df_Y_selec)

    temp_df["mean"] = df_X_selec.rolling(window).mean()
    temp_df["min"]=df_X_selec.rolling(window).min()
    temp_df["max"]=df_X_selec.rolling(window).max()
    temp_df["median"] = df_X_selec.rolling(window).median()
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

for n in range(0,final_df.shape[0]//n_samples_device_window):
    df_X_selec = df_X[n * n_samples_device_window:n * n_samples_device_window + n_samples_device_window]
    df_Y_selec = df_Y[n * n_samples_device_window:n * n_samples_device_window + n_samples_device_window]

    selected_device=df_Y_selec.iloc[0,0]

    print(selected_device)
    labels.append(selected_device)

    X_train, X_test = train_test_split(df_X_selec, test_size=test_size)

    # scaler = QuantileTransformer(n_quantiles=100,random_state=42)
    # scaler=MinMaxScaler()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = KNN(contamination=contamination, n_neighbors=5)

    print(X_train.shape)
    rf.fit(X_train)

    pred = rf.predict(X_train)
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    print("\t", unique_elements, "    ", counts_elements)
    if counts_elements.shape[0] == 2:
        list_train.append(counts_elements[0] * (1 / (1 - test_size)))
    else:
        list_train.append(n_samples_device_window)

    pred = rf.predict(X_test)
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    print("\t", unique_elements, "    ", counts_elements)
    if counts_elements.shape[0] == 2:
        list_test.append(counts_elements[0] * (1 / test_size))
    else:
        list_test.append(n_samples_device_window)

    max_FP = 0
    min_TP = 50000
    for adv in range(0, final_df.shape[0] // n_samples_device_window):
        if (adv == n):
            continue
        dev_x_adv = df_X[adv * n_samples_device_window:adv * n_samples_device_window + n_samples_device_window]
        dev_y_adv = df_Y[adv * n_samples_device_window:adv * n_samples_device_window + n_samples_device_window]

        dev_x_adv = scaler.transform(dev_x_adv)

        res_others = rf.predict(dev_x_adv)
        unique_elements_others, counts_elements_others = np.unique(res_others, return_counts=True)
        if counts_elements_others.shape[0]==2:
            if(dev_y_adv.iloc[0,0]!=df_Y_selec.iloc[0,0]):
                if counts_elements_others[0]>=max_FP:
                    max_FP = counts_elements_others[0]
            if(dev_y_adv.iloc[0,0]==df_Y_selec.iloc[0,0]):
                if counts_elements_others[0]<=min_TP:
                    min_TP = counts_elements_others[0]
        print("\tAdv:",dev_y_adv.iloc[0,0],": ",unique_elements_others,"    ",counts_elements_others)
    if(min_TP==50000):
        min_TP=0
    list_FP.append(max_FP)
    list_TP.append(min_TP)

list_TP=[x/n_samples_device_window for x in list_TP]
list_FP=[x/n_samples_device_window for x in list_FP]
list_train=[x/n_samples_device_window for x in list_train]
list_test=[x/n_samples_device_window for x in list_test]


fig,ax = plt.subplots()
fig.set_size_inches(10.5, 7.5, forward=True)

plt.xticks(np.arange(1,final_df.shape[0]//n_samples_device_window+1, 1.0),labels,rotation='vertical')

ax.plot(range(1,final_df.shape[0]//n_samples_device_window+1), list_train, color="yellow", marker="o")
ax.plot(range(1,final_df.shape[0]//n_samples_device_window+1), list_test, color="orange", marker="o")
#ax.plot(range(1,final_df.shape[0]//n_samples_device_window+1), list_fing, color="green", marker="o")
ax.plot(range(1,final_df.shape[0]//n_samples_device_window+1), list_TP, color="red", marker="o")
ax.plot(range(1,final_df.shape[0]//n_samples_device_window+1), list_FP,color="blue",marker="o")
ax.set_xlabel("device id",fontsize=14)

ax.set_ylim([0,1.05])
plt.legend(['Train','Test','TP Fing','Min TP', 'Max FP'], loc='upper left')

plt.show()
