import os
import numpy as np 
import IPython.display as ipd
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


simple_rate, data = wavfile.read('ornekler/wow/6982fc2a_nohash_0.wav')
train_audio_path="./ornekler/"

samples, sample_rate = librosa.load(train_audio_path+'wow/6982fc2a_nohash_0.wav', sr = 16000)
ipd.Audio(samples, rate=sample_rate)
print(sample_rate)

samples = librosa.resample(samples, sample_rate, 8000) 
ipd.Audio(samples, rate=8000) # 8k lı örnek ses

labels=["bed","bird","five","wow"]


no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    

#kaçar örnek var
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
#plt.show()


#ses kayıtlarının uzunlukları
duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
#plt.hist(np.array(duration_of_recordings))


train_audio_path = './ornekler'
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
            
            
            
#  encode 
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)
print(y)


all_wave = np.array(all_wave).reshape(-1,8000,1)
print(all_wave.shape)
waves=np.array(all_wave).reshape(-1,8000)
print(waves.shape)


#plt.plot(waves,'r--',  label='line 2', linewidth=2,markersize=12)
#plt.show()

len(all_label)