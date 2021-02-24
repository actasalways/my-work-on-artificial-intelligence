# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import numpy as np
import pandas as pd
from scipy.fft import fft

#ses = "0f7dc557_nohash_1.wav"
ses="deneme123.wav"
x,sr = librosa.load(ses)
# x -> ses zaman serisi
# sr -> ses frekansı(Hz)
print(x)
print(x.shape)

print(sr)

dat= np.fft.fft(x)
print(x)
###

a = np.fft.fft(x)
t = np.linspace(0, 0.5, 500)
T = t[400] - t[1]  # sampling interval 
N = x.size
print(t)


plt.plot(a)
plt.grid()

plt.show()






