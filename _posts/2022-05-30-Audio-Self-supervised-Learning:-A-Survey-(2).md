---
#layout: post
title: "Audio Self-supervised Learning: A Survey (2) Audio"
#subtitle: "[Tips]"
date: 2022-05-30 14:22
#background: 
tag: [Self-Supervised Learning, SSL, Audio]
#classes: wide
published: false

toc: true
toc_sticky: false
---

분류 방식은 논문에는 Contrastive Model 중에서 Auto-Encoding/Siamese/Clusetering 그리고 Contrastive Model로만 분류가 되어 있습니다.  
세부적인 분류는 "Audio Self-supervised Learning: A Survey" 설명만 읽고 임의로 분류했기에 분류가 틀릴 수도 있습니다.

https://colorhunt.co/palette/f9ebc8fefbe7dae5d0a0bcc2  
<mark style='background-color: #A0BCC2'>**더 찾아봐야할 부분들**</mark>  
<br>

# 1. Predictive Models
## 1) Auto-Encoder
### **a) Word2Vec: CBow & Skip-gram**
1. Audio2Vec & Speech2Vec
    - CBoW는 이전과 이후의 frame들을 가지고 가운데 비어있는 spectrogram frame 복원
    - CBoW는 acoustic scene classification에서 좋은 결과
    - Skip-gram은 주어진 frame으로 이전과 이후 frame 예측
    - | **Audio2Vec** | **Speech2Vec** |
      | :---: | :---: |
      | explicit한 도움 필요 없음 (supervision한 부분 완전 제거) | 각 단어에 맞는 audio slice segmentation을 위한  <br/> explicit forced alignment technique 사용|
      | CNN 기반 | RNN 기반 |
      | MFCC | Mel-spectrogram |
      | <mark style='background-color: #A0BCC2'>TemporalGap</mark> <br/> 같은 audio clip 내에 random하게 sampled된 data 사이의 정재적 시간 차이 예측 | - |
    - TemporalGap이 CBoW나 Skip-gram보다 더 좋은 결과를 내진 않았지만, pretext task를 상대적인 시간 측정이라는 새로운 관점 제시
2. Carr et al.
    - audio patch suffle >> 다시 순서를 맞추는 방법 (permutation & jigsaw puzzle)
    - "Shuffle and Learn" 논문에서 아이디어 얻음 (audio classification 관련 논문)
    - <mark style='background-color: #A0BCC2'>end-to-end 학습을 위해 재정렬하는 방식을 개선: differential ranking(어떤 건지 확인 더 필요)<mark/>

### **b) Auto-regressive Predictive Coding (APC)**

### **c) Masked Predictive Coding (MPC)**
1. Masked Acoustic Model (MAM)
    - audio input의 일부분을 masking한 후, 전체 input reconsturct
    - reconstruction error 최소화   
2. Mockingjay  
    <br/>  
    <p align="center">
    <img src="../assets/images/Audio-Self-supervised-Learning-A-Survey-(2)/mockingjay.png">
    </p>
    - Mel-Spectrogram
    - random making input을 transformer를 사용하여 coding
    - projection(2 layer MLP + layer normalization)한 후에 frame 예측에 사용 
    - transformer와 projection layer은 동시에 L1 reconstruction loss 최소화
    - transformer의 self-attention의 효과에 대한 연구와 visualization tool 제작  
3. Audio ALBERT
    - Mockingjay와 똑같은 구조
    - trandformer encoder layer parameter 값 동유
        - 빠른 추리(inference), 빠른 학습 속도
        - performance 유지: speaker classificaion과 phoneme(음소: ㄱ,ㄴ,ㄷ,ㅏ,ㅓ,ㅗ ...) classificaion  
4. TERA (Transformer Encoder Representations from Alteration)
    - continuous >> randomness segments
    - channel 측 방향으로 masking (특정 frequency all zero)
    - Gaussian noise 추가
    - 2.Mockingjay와 3.Audio ALBERT보다 좋은 결과
        - performance 향상: speaker classificaion, phoneme classificaion, keyword spotting
        - ASR task에서 기대해 볼 만한 성능: Librispeech, TIMIT dataset
    - [TERA link](https://arxiv.org/pdf/2007.06028.pdf)  

### **d) Non-Auto-regressive Predictive Coding (NPC)**
1. DAPC
    <br/>  
    <p align="center">
    <img src="../assets/images/Audio-Self-supervised-Learning-A-Survey-(2)/DAPC.png">
    </p>  

    - time뿐만 아니라 frequency도 함께 masking
    - 전체가 아닌 mask 부분만 reconstruction >> L1 reconstruction error 최소화
    - CBoW의 확장
    - SpecAugment로 쉽게 만들 수 있음 


## 2) Siamese
### **a) BYOL**
1. BYOL-A
    - 하나의 audio로 negative한 sample 없이 학습
    - log mel-filterbank

# 2. Contrastive Models
### **a) SimCLR approch**
1. LIM Model
    - raw waveform 사용
    - 깉은 utterance에서 나온 chunk들의 encoded representation 최대화  
2. COLA & Fonseca et al. 
    - time-frequency(spectrogram 형태) feature에서 시간(temporal) 축으로 positive sampling
    - patch에 data augmentation 
        - random size cropping
        - Gaussian noise addition
        - <mark style='background-color: #A0BCC2'>mix-back (incoming patch + background patch)</mark>
3. CLAR
    - raw waveform이랑 time-frequency feature에 data augmentation 
    - <mark style='background-color: #A0BCC2'>다양한 augmentation에 대한 연구 진행</mark>
    - 상당히 적은 수의 labelled data를 사용하여 contrastive loss를 결합하면 SSL만 사용했을 때보다 수렴 속도, representation effectiveness의 개선이 있었음
4. Wang
    - raw waveform과 spectral representation 사이의 호응(agreement) 최대화
    - Audioset, ESC-50 downstream task에 효과적

