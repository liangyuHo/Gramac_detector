#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import math
import networkx as nx
import os
import copy
from collections import Counter
from nltk.util import ngrams
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import ensemble,metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def get_maldata():
    df = pd.read_csv('../train/Mal_Result2.csv',header=None)
    #df = df.drop(columns='CPU',axis=1)
    df = df.replace(np.nan,0)
    df = df.to_numpy()
#     df.drop(df.loc[df[1]==8].index, inplace=True)
    return df
def get_bendata():
    df = pd.read_csv('../train/Ben_Result2.csv',header=None)
    #df = df.drop(columns='CPU',axis=1)
    df = df.replace(np.nan,0)
    df = df.to_numpy()
#     df = df[:40000]
    return df
def feature_trans_type(feature):
    temp = np.zeros((len(feature), 6))
    for i in range(len(feature)):
        feature[i] = feature[i][1:len(feature[i])-1]
        feature[i] = list(feature[i].split(","))
        feature[i] = list(map(float, feature[i]))
        feature[i] = [0 if math.isnan(x) else x for x in feature[i]]
        temp[i] = np.array(feature[i])
    return temp


# In[74]:


mal_data = get_maldata() #拿到malware的資料
ben_data = get_bendata() #拿到bengin的資料

mal_label = mal_data[:,1] #刪除malware的名字
# mal_label = np.full(len(mal_label),1)
ben_label = ben_data[:,1] #刪除bengin的名字
label = np.concatenate((mal_label,ben_label),axis=0) #接在一起

mal_feature = mal_data[:,2:] #刪除malware的名字和label
ben_feature = ben_data[:,2:] #刪除benign的名字和label

feature = np.concatenate((mal_feature,ben_feature),axis=0) #接在一起
# feature = np.delete(feature, np.s_[3], axis=1)

#正規化
for i in range(feature.shape[1]):
    v = feature[:,i]
    normalized_v = v/np.linalg.norm(v)
    feature[:,i] = normalized_v
#     print(normalized_v)
#     print(max(normalized_v))

#對label 做one hot encoding
from sklearn.preprocessing import OneHotEncoder
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)
label = label.reshape(-1,1)

onehotencoder = OneHotEncoder()
data_str_ohe =onehotencoder.fit_transform(label).toarray()
label = pd.DataFrame(data_str_ohe)


# In[79]:


X_train,X_test,Y_train,Y_test = train_test_split(feature,label,test_size=0.2)
X_train.shape


# # RF

# In[80]:


#RF
time_start = time.time()
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(X_train, Y_train)
test_y_predicted = forest.predict(X_test)    
print('RF',accuracy_score(Y_test,test_y_predicted))
time_end = time.time()   
time_c= time_end - time_start  
print('time cost', time_c, 's')


# In[88]:


import pickle
pickle.dump(forest_fit,open("RandomForest.pickle","wb"))


# # KNN

# In[89]:


#KNN 
time_start = time.time()
knn = KNeighborsClassifier(weights='distance',n_neighbors=5)
knn_fit = knn.fit(X_train, Y_train)
y_predict = knn.predict(X_test)
print('KNN',accuracy_score(Y_test,y_predict))
time_end = time.time()    
time_c= time_end - time_start   
print('time cost', time_c, 's')


# In[90]:


import pickle
pickle.dump(knn_fit,open("KNN.pickle","wb"))


# # XGboost

# In[78]:


from xgboost import XGBClassifier
time_start = time.time()
xgboostModel = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
xgboostModel.fit(X_train, Y_train)
y__predict = xgboostModel.predict(X_train)
time_end = time.time()  
print(y__predict)
print('xgboost',accuracy_score(Y_train,y__predict))
time_c= time_end - time_start
print('time cost', time_c, 's')


# In[92]:


import pickle
pickle.dump(xgboostModel,open("xgboost.pickle","wb"))


# # MLP

# In[85]:


from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Dropout

X_train = np.asarray(X_train).astype('float32')
Y_train = np.asarray(Y_train).astype('float32')
X_test = np.asarray(X_train).astype('float32')
Y_test = np.asarray(Y_train).astype('float32')

model = Sequential()
model.add(Dense(input_dim=7,units=128,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(units=256,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(units=128,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))
model.summary()


# In[86]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=X_train,y=Y_train,epochs=30,batch_size=32)
result = model.evaluate(X_test,Y_test)
print('\n Train ACC:',result[1])


# In[ ]:


# import matplotlib.pyplot as plt
# k_range = range(1,10)  
# k_scores = []
# for k in k_range:
#     neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
#     neigh.fit(X_train, Y_train)
#     y__result = neigh.predict(X_test) 
#     k_scores.append(accuracy_score(Y_test, y__result))
#     print('neighbors: ', k, 'score: ', accuracy_score(Y_test, y__result))
    


# In[ ]:


# plt.plot(k_range,k_scores)
# plt.xlabel('Value of K')
# plt.ylabel('Accuracy')
# plt.show()


# In[ ]:




