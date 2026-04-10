import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("Dataset/MoneyLaunderingdata.csv")
dataset.fillna(0, inplace = True)
cols = ['type','nameOrig','nameDest']
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
dataset[cols[2]] = pd.Series(le3.fit_transform(dataset[cols[2]].astype(str)))

Y = dataset['isFraud'].ravel()
dataset.drop(['isFraud'], axis = 1,inplace=True)
X = dataset.values

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

cls = RandomForestClassifier(ccp_alpha=0.5)
cls.fit(X_train, y_train)
predict = cls.predict(X_test) 

f = f1_score(y_test, predict,average='macro') * 100
print(f)

CM = confusion_matrix(y_test, predict)

TN = CM[0][0] / len(y_test)
FN = CM[1][0] / len(y_test)
TP = CM[1][1] / len(y_test)
FP = CM[0][1] / len(y_test)
print("TN "+str(TN)+" FN "+str(FN)+" TP "+str(TP)+" FP "+str(FP))

fft_data = fft(X)

X = []
for i in range(len(fft_data)):
    temp = []
    for j in range(len(fft_data[i])):
        temp.append(float(fft_data[i,j]))
    X.append(temp)
X = np.asarray(X)    
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

cls = RandomForestClassifier()
cls.fit(X_train, y_train)
predict = cls.predict(X_test) 

f = f1_score(y_test, predict,average='macro') * 100
print(f)

CM = confusion_matrix(y_test, predict)

TN = CM[0][0] / len(y_test)
FN = CM[1][0] / len(y_test)
TP = CM[1][1] / len(y_test)
FP = CM[0][1] / len(y_test)
print("TN "+str(TN)+" FN "+str(FN)+" TP "+str(TP)+" FP "+str(FP))













