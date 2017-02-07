#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 08:55:32 2017

@author: haldun
"""

#perceptron example

import numpy as np

datalen = 10 # her bir sınıftan alınacak veri sayısı
shift = 1 # 0.5 #ilk küme için merkez ötelemesi

ta = np.random.rand(datalen,2) #veri kümesi 1 oluşturulur
ta = ta + shift # veri kümesi 1 merkezi shift+0.5,shift+0.5 olan gürültülü bir veridir
tal = np.ones((datalen,1)) # veri kümesi 1 için etiketler oluşturulur
da = np.append(ta, tal, axis = 1) # etiketler ile birlikte bir tam veri seti oluşturulur

tb = np.random.rand(datalen,2) #veri kümesi 2 oluşturulur bu veri merkezi 0.5,0.5 olan gürültülü bir veridir 
tbl = -np.ones((datalen,1)) # veri kümesinin etiketi -1 olarak oluşturulur
db = np.append(tb, tbl, axis = 1) # etiketler ile birlikte bir tam veri seti oluşturulur

dlearn = np.append(da,db,axis=0) # iki veri seti öğrenme kümesi oluşturmak üzere birleştirlir.

# burada veri setini oluştururken etiketleri de dahil etmem tamamen benim tercih ettiğim bişi takibi kolay olsun diye yaptım
# fakat aynı zamanda kafa karışıklığına sebep olabildiği yerler de mevcut bu durum için farklı yaklaşımlar geliştirilebilir


learnrate = 0.02 # güncelleme ağırlığı/ öğrenme oranı

w = np.ones((3,1))*0.5#np.random.rand(3,1) # ağırlık 1

errate = 0.01 # hata oranı için çıkış limiti - bu oranda bir doğru sınıflama yeterli olacaktır 
epochlim = 100 # eğitim tekrarlama limiti - uzun süren bir eğitim yetersiz olabilir aynı zamanda sonsuz döngüde kalmamamız önemli
epoch = 0 # eğitim tekrar sayısı
err = 1 # hesaplanan hata oranı
errl = np.zeros((len(dlearn),1)) #öğrenme matrisi


def datacalc(w, learn): # anlık ağırlıklara göre sınıf hesaplanır 
    return np.sign(w[0] * learn[0] + w[1]*learn[1] + w[2])

def w_update( m, learn): # güncelleme fonksiyonu
    #inLearn = np.append(learn(0:3),1)
    # w = w + m*inLearn*learnrate
    w[0] = w[0] + m * learn[0] * learnrate 
    w[1] = w[1] + m * learn[1] * learnrate 
    w[2] = w[2] + m * 1 * learnrate

def w_classify(w,data):#shift) #np.array şeklinde verili bir değerin sınıf değerini döndürür
    #return np.sign(np.dot(np.append(np.random.rand(1,2)-shift,np.array(1)),w))
    ##calculateddata = w[0]*data[0] + w[1]*data[1] + w[2]
    datainner = np.append(data,np.array(1)) # x + y + (1 -> c ile denk gelmesi için bir ekleme yapıyoruz 
    calculateddata = np.dot(datainner,w) # w_ax + w_by + w_c
    return np.sign(calculateddata)
    

while err > errate and epoch < epochlim : # öğrenme döngüsü
    for i in range(len(dlearn)):
        m = dlearn[i,2] - datacalc(w,  dlearn[i]) # istenen sınıf ve tespit edilen sınıf farklı ise buna göre işlem yapılır
        w_update(m, dlearn[i])
        errl[i] = m 

    err = pow(pow(errl,2).sum(),0.5)/len(dlearn)
    print errl.T, epoch
    epoch +=1