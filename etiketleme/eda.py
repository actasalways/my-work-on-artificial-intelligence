import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa 

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,sharey=True, figsize=(20,5))
    fig.suptitle('Time Series(Zaman Serisi)', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Dönüşümü', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Katsayıları', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Katsayıları', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def esik_gec(y,rate, esik_degeri):
    gecti_mi =[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True ).mean()
    for mean in y_mean:
        if mean > esik_degeri:
            gecti_mi.append(True)
        else:
            gecti_mi.append(False)
    return gecti_mi 
            

def calc_fft(y,rate):
    n=len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y=abs( np.fft.rfft(y)/n)
    return (Y,freq)

df= pd.read_csv('instruments.csv')
df.set_index('fname',inplace=True)

for f in df.index:
    rate,signal = wavfile.read('wavfiles/'+f)
    df.at[f,'length'] = signal.shape[0] /rate 

classes=list(np.unique(df.label))
class_dist=df.groupby(['label'])['length'].mean()

fig,ax = plt.subplots()
ax.set_title('Sınıf Dağılımı',y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal') 
plt.show()


df.reset_index(inplace=True)

signals={}
fft={}
fbank={}
mfccs={}


for c in classes:
    wav_file=df[df.label==c].iloc[0,0]
    signal, rate = librosa.load('wavfiles/'+wav_file, sr=44100 )
    esik_gecildi_mi=esik_gec(signal,rate,0.0005)
    signal=signal[esik_gecildi_mi]
    signals[c]=signal
    fft[c]=calc_fft(signal,rate )   

    bank=logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T #→ nfft 44100/40  
    fbank[c]=bank
    
    mel=mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft= 1103).T
    mfccs[c]=mel

plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()
plot_fbank(fbank)
plt.show()
plot_mfccs(mfccs)
plt.show()


"""
if len(os.listdir('clean')) == 0 :
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/'+f,sr=16000)
        gecti_mi = esik_gec(signal,rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[gecti_mi])
"""












