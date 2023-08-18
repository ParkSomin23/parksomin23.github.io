---
#layout: posts
title: "Learning Filterbanks From Raw Speech For Phone Recognition"
date: 2022-08-02 18:10
category: "논문-리뷰"
tag: [Audio, Wav2Vec, Filter]
published: true

toc: true
toc_sticky: true

use_math: true
---

논문: [Learning Filterbanks From Raw Speech For Phone Recognition](https://arxiv.org/pdf/1711.01161.pdf)  
연관 포스트: 
1. [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations]({% post_url 2022-08-03-Wav2Vec-network %})

> **Abstract**
> 1. bank of complex filters를 학습하여 end-to-end phone recognition
> 2. Time-Domain(TD) filterbanks가 mel-filterbank보다 성능이 좋음
> 3. 수렴한 TD-filers들은 asymmetric(비대칭적) impulse respose를 가지고 있으며, 그 중 일부는 analytic하게 남아 있었음

# 1. Introduction
- Speech Features의 대표적인 표현법은 [gammatones](https://en.wikipedia.org/wiki/Gammatone_filter)나 Mel-filterbanks(mel frequency spectral coefficient, MFSC)
- 다른 논문보다 lightweight한 구조, deep learning에서도 사용가능한 learnable MFSC
- complex convolution weight를 MFSC의 center-frequency와 bandwith와 일치하는 Gabor wavelets으로 initalize하기에 pretraining도 필요 없음

# 2. Time-Domain MFSC
## 1) MFSC Computation
- input signal $x \rightarrow \text{STFT} \rightarrow \text{mel filter(frequency average)}$
- $Mx(t,n)=\frac{1}{2\pi}\int\vert\hat{x}_t(\omega)\vert^2\vert\hat{\psi}_n(\omega)\vert^2 d\omega$ 
    - $\phi$: Hanning window fo width $s$
    - $(\psi_n)_{n=1..N}$: N filter with squared frequency response triangle
        - cetered on $(\eta_n)_{n=1..N}$
        - full width at half maximum(FWHM) $(\omega_n)_{n=1..N}$
    - $x_t: u \mapsto x(u)\phi(t-u)$ : windowed signal at time step $t$
    - $\hat{f}$: Fourier transform of function $f$
    - $(t \mapsto Mx(t,n))_{n=1..N}$: filterbank set of N functions  
<br/>

## 2) Approximating MFSC with Convolutions in Time 

<p align="center">
<img src="../assets/images/TD_Filter/fig1.png" width="40%">
<em><br/>MFSC와 Garbor로 근사한 필터를 이용한 spectrogram</em>
</p>  
<br/>

- $Mx(t,n) \approx \vert x \ast \varphi_n \vert^2 \ast \vert\phi\vert^2(t)$
- $\varphi_n$: n-th triangular filter in frequency approximated wavelet 
    - $\vert \hat{\varphi}_n\vert^2 \approx \vert\hat{\psi}_n\vert^2$
    - $\varphi_n$ 이 Hanning window $\phi$ 보다 작으면 근사값 성립됨  
- Initialization은 Garbor Filter로 MFSC에 근사하도록 만듦  
    - $\varphi_n(t) \propto e^{-2\pi i \eta_n t}\frac{1}{\sqrt{2\pi}\sigma_n} e^{-\frac{t^2}{2\sigma^2_n}}$  
    - frequency $\xi$ 는 $\hat{\varphi}_n(\xi)\propto \sqrt{\sigma_n}e^{-\frac{1}{2}\sigma^2_n(\xi-\eta_n)^2}$,  
  FWHM은 $2\sqrt{2\text{log}2}\sigma_n^{-1}$ 로, $\quad \sigma_n=\frac{2\sqrt{2\text{log}2}}{\omega_n}$
    - $\varphi_n$ 를 normalize해서 $\psi_n$ 과 에너지가 같도록 만듦  
    <br/>

- **\<Gabor 추가 설명>**
    - Gabor Filter는 scale, time, frequency, phase로 wave를 표현할 수 있음
    <p align="center">
    <img src="../assets/images/TD_Filter/gabor.png" width="100%">
    <em><br/>Gabor Filter 구현 코드</em>
    </p>  
    <p align="center">
    <img src="../assets/images/TD_Filter/모두연.png" width="100%">
    <em><br/>Gabor Filter 값 예시(LEAF 포스트)</em>
    </p>  

## 3) Network
<p align="center">
<img src="../assets/images/TD_Filter/tb1.png" width="60%">
</p>   

| **내용** |  **MFSC** | **Learnable** |
|  :---:  |   :---:   |     :---:     |
|**sample rate**|16kHz|16kHz|
|**Hanning Window**|Hanning window width=400|convolution filter width=400|
|**hop length**|10ms|convolution stride=160|
|**N_Filter**| 40 | 80 (real + imaginary)<br/>L2 pooling + square layer에서 $\sqrt{\text{real}^2 + \text{imag}^2}$ 해서<br/> 40으로 만듦|
|**log**| max(log-spectrogram, 1) | log(abs(group conv.) + 1) |
|**pre-emphasis**|$\alpha$=0.97| no |
|**mean-variance normalization**| yes | no (only on waveform) |  

<br/>

# 3. Experiments
1. Dataset
    - TIMIT phone recognition with 39 phonemes  
    <br/>

2. Model
    - 1\) 5 layer conv. + 1000 feature maps + ReLU + dropout 0.5
    - 2\) dropout 0.7 빼고 위와 같음
    - 3\) 8 layer conv. + PReLU + dropout 0.7  
    <br/>

3. TD-Filer
    - 1\) Fixed: Init filter and fix
    - 2\) Learn-all: Init filter and learn + learn averaging
    - 3\) Learn-filterbank: Init filter and learn + averaging(Hanning window)
    - 4\) Randinit: init radomly and learn

    <p align="center">
    <img src="../assets/images/TD_Filter/tb2.png" width="60%">
    </p>  
    
    - fixed: TD-Filterbanks 전에 mean-variance normalization 없어서 결과가 덜 좋음
    - randinit: 가장 결과가 안 좋음, initalization의 중요성
    - learn-all: averaging filter는 initalization 값과 유사하다고 판단됨
    - learn-filterbank: 최종적으로 적합하다고 생각되는 model  
    <br/>
4. Results
<p align="center">
<img src="../assets/images/TD_Filter/tb3.png" width="80%">
<em><br/>MFSC보다 wav가 결과가 항상 좋다</em>
</p> 

# 4. Analysis of Learned Filters
- 8 layer conv. + PReLU + dropout 0.7 model에서 학습된 filer 분석
<p align="center">
<img src="../assets/images/TD_Filter/fig2.png" width="80%">
</p> 

<p align="center">
<img src="../assets/images/TD_Filter/fig3.png" width="80%">
</p> 

- inital filter 모양은 잘 유지하면서도, filter bandwidth의 variability 생김
- asymmetric한 형태로 학습되는데 사람의 청각 시스템도 asymmetric
- inital filter 모양을 유지하는 이유?
    - 성능 향상을 위한 complex filter의 fully generality가 필수적이지 않음
    - analytic fliter가 real-domain에서 신호처리하는 것과 유사
- analytic signal은 negative frequency에 energy가 없어야함
    - 하지만 learned filter에서 negative한 값들도 존재
    - $\frac{\text{negative energy}}{\text{positive energy}}$에서 0(analytic) ~ 1(pure real filter)라고 할 때, 평균 값은 0.26
    - negative한 energy가 많은 주파수 대역은 1000~3000 Hz 사이
    - positive frequency의 down-scaled되어 있음