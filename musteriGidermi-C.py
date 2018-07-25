# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 19:29:23 2018

@author: ahmet
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#veriyi böldük gereksiz kısımları(ID,soyad,rowNumber)çıkardık -- neye bakıp , neyi öğrenecek [X-->Y]
X = veriler.iloc[:,3:13].values
Y = veriler.iloc [:,13].values 

#encoder:  Kategorik -> Numeric 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1]= le.fit_transform(X[:,1])# burada ülkeleri 0-1-2 diye grupladık
X[:,2]= le.fit_transform(X[:,2])# burda cinsiyei 0-1 diye ayarladık

from sklearn.preprocessing import OneHotEncoder # 0-1-2 yerine matrixe ayırıyoruz
ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
'''
001
100
010
001  gibi bir hale dönüştürdük
'''
X = X[:,1:] # burada ilk satırı attık ( Dummy Variable ) çünkü 3 ülke var 2 ülkeden biri değilse demekki 3. ülke

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0) #veriyi test ve eğitim için ayırdık

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# veri içerisindeki sayıları 0-1 arasına indiriyoruz YSA'ya girdi olarak verebilmek için
# bu işlem daha düzgün de (başka yollarla) yapılabilir
X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test)
# büyük harf ile olanlar scale edilmiş veriler artık

#yapay sinir ağı
import keras
from keras.models import Sequential # yapay sinir ağı modeli
from keras.layers import Dense #katmanların inşası için

classifier = Sequential() #bu ifadeden sonra Ram üzerinde artık bir sinir ağı bizi bekliyor
#6'lı bir gizli katman olacak
#ilk katsayılar uniform olarak rasgele verilecek
#activation function olarak relu kullanacak
# ilk katman-giriş katmanında 11 adet nöron olacak
classifier.add(Dense(6,init='uniform',activation='relu',input_dim=11)) # giriş katmanı
classifier.add(Dense(6,init='uniform',activation='relu')) # hidden layer --gizli katman
classifier.add(Dense(1,init='uniform',activation='sigmoid')) # çıkış katmanı 

# adam -> sinapsisin katsayılarını nasıl optimize edeceğimizi ifade ediyor  
# binary_crossentropy -> sonuç binary (0-1) olduğu için bunu seçtik
# ölçü yani metrics olarak accuracy seçtik 1'ler 1 ise doğru 0'lar 0 ise doğru gibi düz mantık HERALDE !!!
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,epochs=1000) # eğit
y_pred = classifier.predict(X_test) # X_train verisine bakarak sonuçları tahmin et

#y_pred şuan için 0-1 arası ondalık sayılar bunu biz 0-1 yapacaz yani müşteri gider mi gitmez mi
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) # gerçek sonuçlar ile tahmin edilen sonuçları karşılaştır
print(cm)





















