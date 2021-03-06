# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import IPython.display as ipd
import librosa
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


simple_rate, data = wavfile.read('../ornekler/wow/6982fc2a_nohash_0.wav')
train_audio_path="../ses_dosyalari/"

samples, sample_rate = librosa.load(train_audio_path+'wow/6982fc2a_nohash_0.wav', sr = 16000)
ipd.Audio(samples, rate=sample_rate)
print(sample_rate)

samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000) # 8k lı örnek ses

labels=["bed","bird"]
    

train_audio_path = '../ses_dosyalari'
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)
            
            
# satırların encode edilmesi
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)
print(y)


all_wave = np.array(all_wave).reshape(-1,8000,1)
print(all_wave.shape)
waves=np.array(all_wave).reshape(-1,8000)
print(waves.shape)


x_train, x_test, y_train, y_test = train_test_split(np.array(waves),np.array(y),stratify=y,test_size = 0.33,random_state=7,shuffle=True)
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

###########################################################################################

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(x_train)
X_test2 = pca.transform(x_test)



from sklearn.svm import SVC
svc = SVC(kernel='poly',gamma='scale')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)