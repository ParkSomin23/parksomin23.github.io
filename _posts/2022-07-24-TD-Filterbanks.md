---
#layout: posts
title: "Learning Filterbanks From Raw Speech For Phone Recognition"
date: 2022-07-24 20:08
category: "논문-리뷰"
tag: [Audio, Wav2Vec]
published: true

toc: true
toc_sticky: false

use_math: true
---

논문: [Learning Filterbanks From Raw Speech For Phone Recognition](https://arxiv.org/pdf/1711.01161.pdf)  
연관 포스트: 
1. (예정)[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations]
2. (예정)[Categorical Reparameterization With Gumbel-Softmax]

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
- 수식: 
    - $\phi$: Hanning window fo width $s$
    - $(\psi_n)_{n=1..N}$: N filter with squared frequency response triangle
        - cetered on $(\eta_n)_{n=1..N}$
        - full width at half maximum(FWHM) $(\omega_n)_{n=1..N}$
    - $x_t: u \mapsto x(u)\phi(t-u)$ : windowed signal at time step $t$
    - $\hat{f}$: Fourier transform of function $f$
    - $(t \mapsto Mx(t,n))_{n=1..N}$: filterbank set of N functions
    $$
    \quad Mx(t,n)=\frac{1}{2\pi}\int\vert\hat{x}_t(\omega)\vert^2\vert\hat{\psi}_n(\omega)\vert^2 d\omega
    $$  
<br/>

## 2) Approximating MFSC with Convolutions in Time
$$
Mx(t,n) \approx \vert x \ast \varphi_n \vert^2 \ast \vert\phi\vert^2(t)
$$
- $\varphi_n$: n-th triangular filter in frequency approximated wavelet 
    - $\vert \hat{\varphi}_n\vert^2 \approx \vert\hat{\psi}_n\vert^2$
- $\varphi_n$ 이 Hanning window $\phi$ 보다 작으면 근사값 성립됨

