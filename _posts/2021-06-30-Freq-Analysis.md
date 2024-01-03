---
#layout: posts
title: "Frequency Analysis"
date: 2021-06-30 13:18
category: "정리함"
tag: ["Audio"]

published: true
toc: true
toc_sticky: true
---

> google drive[[link](https://drive.google.com/file/d/1YAvTa8HlMgQf3q8oXBTYFduPpghGM5FX/view?usp=share_link)]에서 해당 음성들을 들을 수 있습니다<br>
현재 keynote 형식으로만 지원되고 있습니다

# 1. PPT
<p img-align="center">
<img src="../Freq-Analysis/img_0.png" width="100%">
<img src="../Freq-Analysis/img_1.png" width="100%">
<img src="../Freq-Analysis/img_2.png" width="100%">
<img src="../Freq-Analysis/img_3.png" width="100%">
<em>해당 음성은 저작권과 관련하여 소리를 들을 수 없습니다</em>
<img src="../Freq-Analysis/img_4.png" width="100%">
<em>해당 음성은 저작권과 관련하여 소리를 들을 수 없습니다</em>
<img src="../Freq-Analysis/img_5.png" width="100%">
<img src="../Freq-Analysis/img_6.png" width="100%">
<img src="../Freq-Analysis/img_7.png" width="100%">
<em>해당 음성은 google drive에서 확인하세요<br>
사용된 데이터셋: https://github.com/CheyneyComputerScience/CREMA-D</em>
<img src="../Freq-Analysis/img_8.png" width="100%">
<em>해당 음성은 google drive에서 확인하세요<br>
사용된 데이터셋: https://github.com/CheyneyComputerScience/CREMA-D</em>
<img src="../Freq-Analysis/img_9.png" width="100%">
<img src="../Freq-Analysis/img_10.png" width="100%">
<img src="../Freq-Analysis/img_11.png" width="100%">
</p>

# 2. Code
```python
import os
import numpy as np

from scipy import signal
from scipy.signal import butter

import scipy.io
from scipy.io import wavfile

import IPython.display as ipd
import matplotlib.pyplot as plt

import librosa
```


```python
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = signal.lfilter(b, a, data)
    
    return y

def lowpass_filter(data, highcut, fs, order=5):
    
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low', analog=False)
    
    y = signal.lfilter(b, a, data)
    
    return y

def highpass_filter(data, lowcut, fs, order=5):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high', analog=False)
    
    y = signal.lfilter(b, a, data)
    
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
```
```python
lowcut = 400
highcut= 2000
fs = 16000
for order in [3, 5, 7]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = signal.freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
```
    
![png](../Freq-Analysis/output_2_1.png)
    

```python
path = 'engInNY.wav'
sr, y = wavfile.read(path)

time = np.linspace(0, len(y)/sr, y.shape[0])
plt.plot(time, y[:, 0])
plt.plot(time, y[:, 1])
plt.show()
```
    
![png](../Freq-Analysis/output_4_0.png)
    

```python
ipd.Audio(y[:, 0], rate=sr)
```

```python
mix = y[:, 0]*0.5 + y[:, 1]*0.5
mix = mix[int(sr*8):]
order=5
time = np.linspace(0, len(mix)/sr, mix.shape[0])
ipd.Audio(mix, rate=sr)
```

```python
N = len(mix)

k = np.arange(N)
T = N / sr
freq = k / T
freq = freq[range(int(N/2))]

# FFT 적용
yfft = np.fft.fft(mix)
yf = yfft / N
yf = yf[range(int(N/2))]

plt.rcParams["figure.figsize"] = (15,4)

# FFT 출력
plt.plot(freq, abs(yf), 'b')
plt.xlabel('Frequency')

plt.ylabel('Amplitude')
plt.xlim(0, sr/2)

plt.plot([70, 70], [0, 500])
plt.plot([150,150], [0, 500])
plt.plot([300,300], [0, 500])
plt.plot([400,400], [0, 500])
plt.plot([800,800], [0, 500])
plt.plot([1500,1500], [0, 500])
plt.plot([3000,3000], [0, 500])
plt.plot([6000,6000], [0, 500])
plt.plot([12000,12000], [0, 500])
plt.show()

plt.plot(freq, abs(yf), 'b')
plt.xlabel('Frequency')

plt.ylabel('Amplitude')
plt.xlim(0, sr/20)

plt.plot([70, 70], [0, 500])
plt.plot([150,150], [0, 500])
plt.plot([300,300], [0, 500])
plt.plot([400,400], [0, 500])
plt.plot([800,800], [0, 500])
plt.plot([1500,1500], [0, 500])
plt.plot([3000,3000], [0, 500])
plt.plot([6000,6000], [0, 500])
plt.plot([12000,12000], [0, 500])
plt.show()
```
    
![png](../Freq-Analysis/output_7_0.png)
![png](../Freq-Analysis/output_7_1.png)
    

```python
#  - 70Hz
bp = lowpass_filter(mix, 70, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_8_0.png)
    

```python
# 70 - 150Hz
bp = bandpass_filter(mix, 70, 150, sr, order=3)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_9_0.png)

```python
# 150 - 300Hz 
bp = bandpass_filter(mix, 150, 300, sr, order=3)
time = np.linspace(0, len(bp)/sr, bp.shape[0])
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_10_0.png)
    
```python
# 300 - 400Hz
bp = bandpass_filter(mix, 300, 400, sr, order=3)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_11_0.png)
    

```python
# 400 - 800Hz
bp = bandpass_filter(mix, 400, 800, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
``` 
![png](../Freq-Analysis/output_12_0.png)
    

```python
# 800 - 1500Hz 
bp = bandpass_filter(mix, 800, 1500, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_13_0.png)
    

```python
# 1500 - 3000Hz 
bp = bandpass_filter(mix, 1500, 3000, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_14_0.png)
    

```python
# 3000 - 6000Hz
print(sr)
bp = bandpass_filter(mix, 3000, 6000, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```    
![png](../Freq-Analysis/output_15_1.png)
    

```python
# 6000 - 12000Hz 
bp = bandpass_filter(mix, 6000, 12000, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_16_0.png)
    

```python
# 12000Hz -
bp = highpass_filter(mix, 12000, sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_17_0.png)
    

```python
filter = [0,70,150,300,400,800,1500,3000,6000,12000]
bp = np.zeros(mix.shape)
for i, f in enumerate(filter):
    if i >=1 and i<4:
        order = 3
    else:
        order =5
    if f == 0:
        bp += lowpass_filter(mix, filter[1], sr, order)
    elif f == 12000:
        bp += highpass_filter(mix, f, sr, order)
    else:
        bp += bandpass_filter(mix, f, filter[i+1], sr, order)
plt.plot(time, bp)
plt.show()
ipd.Audio(bp, rate=sr)
```
![png](../Freq-Analysis/output_18_0.png)
    

```python
M = librosa.feature.melspectrogram(mix, sr, n_fft=int(sr*0.064), hop_length=int(sr*0.032), n_mels=32)

y_inverse = librosa.feature.inverse.mel_to_audio(M)

time = np.linspace(0, len(y_inverse)/sr, y_inverse.shape[0])
plt.plot(time, y_inverse[:,0])
plt.show()
ipd.Audio(y_inverse[:, 0], rate=sr)
```