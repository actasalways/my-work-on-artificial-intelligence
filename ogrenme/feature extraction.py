# -*- coding: utf-8 -*-



# ön işleme sınıf şeysi

import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd



ses = "./new/ornekler/wow/6982fc2a_nohash_0.wav"
x,sr = librosa.load(ses)
# x -> ses zaman serisi
# sr -> ses frekansı(Hz)

print("x type:",type(x),"sr type",type(sr))
print(x.shape,sr)


ipd.Audio(ses)
ipd.Audio(x,rate=sr)  #Bu şekilde de oynatılabiliyor.


x, sr = librosa.load(librosa.util.example_audio_file())
librosa.output.write_wav('ses_kayit.wav', x, sr)


plt.figure(figsize=(10,2))
librosa.display.waveplot(x,sr=sr)

#
X=librosa.stft(x) #stft -> Short-time Fourier transform
Xdb=librosa.amplitude_to_db(abs(X)) #Genlikten Desibel değerine
plt.figure(figsize=(20,8))
librosa.display.specshow(Xdb,sr=sr,x_axis="time",y_axis="hz")
plt.colorbar()

#################################################   Feature Extraction — Öznitelik Çıkarımı   ######################################
#################################################   Feature Extraction — Öznitelik Çıkarımı   ######################################
#################################################   Feature Extraction — Öznitelik Çıkarımı   ######################################

#Mel frekans ölçeği
mfkk=librosa.feature.mfcc(x,sr=sr)
print(mfkk.shape)
#(20, 216)
plt.figure(figsize=(15,6))
librosa.display.specshow(mfkk,x_axis="s")
plt.colorbar()

#Zero crossing rate - bir sinyalin sıfır çizgisinden geçişi yani işaret değişiminin oranıdır.
zero_crossing=librosa.zero_crossings(x)
print(sum(zero_crossing)) #Toplam sıfır geçişi sayısı
 #1908
plt.plot(x[5000:5100])
plt.grid()


# Spectral Centroid -  Spektrumun kütle merkezinin nerede olduğunu gösterir.
spec_cent=librosa.feature.spectral_centroid(x)
print(spec_cent.shape)
#(1, 216)
plt.semilogy(spec_cent.T)
plt.ylabel("Hz")


#Spectral Rolloff - Sinyal şeklinin ölçüsü. Toplam spektral enerjisinin belli bir yüzdesini temsil eder.
spec_roll=librosa.feature.spectral_rolloff(x,sr=sr)
print(spec_roll.shape)
#(1, 216)
plt.semilogy(spec_roll.T,"r")
plt.ylabel("Hz")



#Chroma Frekansı - Spektrum müzikal oktavının 12 farklı yarı tonunu(chroma) temsil eden 12 parçanın belirtildiği ses için güçlü bir sunumudur.
chroma=librosa.feature.chroma_stft(x,sr=sr)
print(chroma.shape)
  #(12, 216)
librosa.display.specshow(chroma,y_axis="chroma",x_axis="time")
plt.colorbar()


# Spektral Bant Genişliği - Ses sinyalinin dalga genişliğinin maksimum tepe noktasının yarısını tanımlar.
spec_band=librosa.feature.spectral_bandwidth(x,sr=sr)
print(spec_band,spec_band.shape)
# [[3485.33953975 1440.92475081 1564.93918585 ... 4094.54535891   
# 4112.77024258 4179.68287521]]
# (1, 2647)


