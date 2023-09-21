---
title: "Unpaired Motion Style Transfer from Video to Animation"
date: 2021-05-20 18:10
category: "논문-리뷰"
tag: [StyleTransfer, Video]
published: true

toc: true
toc_sticky: true
use_math: true
---


논문: [Unpaired Motion Style Transfer from Video to Animation](https://deepmotionediting.github.io/style_transfer)   

<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00001.jpeg" width="100%">
</p> 

# Introduction
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00002.jpeg" width="100%">
</p> 

- style Transfer은 image style transfer부터 발전해옴
- 좌측: image style transfer<br>
    style 이미지 화풍 + content 이미지의 구성 요소 $\rightarrow$ output 이미지
- 우측: motion style transfer<br>
    style motion 느낌 + content motion 행동 $\rightarrow$ output motion
<br><br>


<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00003.jpeg" width="100%">
</p> 

- 위 식으로 표현 가능
    - $m$, $n$: motion content (ex) walking, running
    - $s$, $t$: motion style (ex) angry, happy
- motion은 <b>joint rotations + joint positions</b>으로 표현 가능
    - joint rotations: unit Quarternion로 표현 (4 차원)
    - joint positions: x, y, z (3 차원)
    - root position과 root rotation도 같이 사용
- content $m$ (31 joint rotations) + sytle $t$ (21 joint positions) $\rightarrow$ style transfered motion (31 joint rotations)
- $T$: frame
<br><br>

# Dataset
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00004.jpeg" width="100%">
</p> 

- Xia & 자체 제작 dataset 사용
- 8 frame씩 겹치게 32 frame 길이로 영상 자름
- BVH 파일: hierarchy of Joints 
    - root: hip is Root 
    - root에 이어서 LeftHipJoint, LeftUpLeg, LeftLeg ... 순서로 표현
    - 좌측 하단: 각 joint rotation은 3 channel, root에는 3 channel position이 추가 제공됨
        - 30 개의 joint * 3 channel + 1 개의 root * 6 channel = 96 channels
    - 우측 하단: frame 별 joint의 channel (96 channels)
<br><br>

# Network
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00005.jpeg" width="100%">
</p> 

- 네트워크는 총 4가지 부분으로 구성
    1. Style Motion Representation Encoder
    2. Context Motion Representation Encoder
    3. Motion Translator
    4. Discriminator

## 1. Style Motion Representation Encoder
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00006.jpeg" width="100%">
</p> 

- "1차원 convolution + Leaky ReLU" $\times$ 2 + max-pooling
- input: root를 포함한 21개의 선택된 joint position의 2 & 3 차원 값
    - 3D: 20 * 3 + 3 (root position) + 1 (root rotation) = 64
    - 2D: 20 * 2 + 2 (root position) = 42
    - 2D 값은 3D data를 2d projection한 값
- output: 각각 144 channel의 vector 생성, 2D & 3D vector 평균값 사용
- training: 2D & 3D input
- inference: input 데이터의 종류에 따라 2개의 인코더 중 1개만 사용

## 2. Content Motion Representation Encoder
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00007.jpeg" width="100%">
</p> 

- "1차원 convolution + instance normalization + Leaky ReLU" $\times$ 3 + residual connection
- instance normalization: content code에서 style 제거 할 수 있음
- input: 31개의 joint rotation * 4 (Quaterion) + 3 (root position) + 1 (root rotation) = 128
- output: 144 channel의 vector 생성
- 우측: content code를 PCA를 활용하여 2차원으로 표현
    - (a) style label: 같은 style끼리 cluster되지 않음 (style removed) 
    - (b) phase label: walking sample들의 phase 표현, 같은 content에 대해서 phase는 연속적으로 분포

## 3. Motion Translator
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00008.jpeg" width="100%">
</p> 

- MLP: 144 channel의 style code가 AdaIN에 사용되는 paramter 수의 2배가 되도록 "linear layer + Leaky ReLU" $\times$ 3
    - AdaIN에서 사용될 mean & variance 값으로 사용됨
- Decoder
    - content code에 style code에서 추출된 mean & variance 값을 활용하여 AdaIN 수행
    - 1차원 convolution, Leaky ReLU, residual connection으로 144 channel로 만듦
    - 원래 motion input data의 모양과 같도록 upsample

## 4.Discriminator
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00009.jpeg" width="100%">
</p> 

- 들어온 입력값이 원래 있는 데이터인지 만들어진 데이터인지 구분하는 network
- discriminator 중간에 feature 값을, 마지막은 class (true/false) 값 추출
- 이 둘은 loss 계산에 사용

## Additonal Consideration
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00010.jpeg" width="100%">
</p> 

- Global Velocity Warping
    - motion에 따라 속도 다름
    - joint의 최대 local velocity의 평균값 사용
    - style과 content의 비율로 계산
- Foot Contact
    - global velocity warping 하기 전의 content 입력값의 발이 땅에 닿는 (foot contact) label 값 사용
    - 스케이트 타는 motion 생성 방지
    - 좀비가 걷는 스타일은 이를 무시히는 결과 보임
<br><br>

# Loss  
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00011.jpeg" width="100%">
</p> 

- 총 5개의 loss 고려됨
    1. content consistency loss ($\mathcal{L}_{con}$)
    2. adversarial loss ($\mathcal{L}_{adv}$)
    3. regression loss ($\mathcal{L}_{reg}$)
    4. joint embedding loss ($\mathcal{L}_{joint}$)
    5. style triplet loss ($\mathcal{L}_{trip}$)

## 1. Content Consistency Loss
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00012.jpeg" width="100%">
</p> 

- 같은 sytle을 가진 data 2개 선택 
    - $\mathbb{m}^s, \mathbb{n}^s$: 같은 $s$ style을 가진 motion data
- $m$ content에 $\mathbb{n}^s$의 $s$ style로 style transfer
    - $F(E_C(\mathbb{m}^s)\;\vert\;E_S(\mathbb{n}^s))$
- 같은 style을 활용하여 style transfer했기에 원본  $\mathbb{m}^s$와의 차이가 작아야 함 
    - L1 loss
- content consistency loss만으로도 style transfer가 잘 되나, 세부적인 부분에서 손실된 부분 존재

## 2. Adversarial Loss
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00013.jpeg" width="100%">
</p> 

- generated data를 진짜 데이터와 구분할 수 없도록 학습
- real data: 1, fake/generated data: 0
- Discriminator의 output 값 사용
- $D^t(\mathbb{n}^t)$ : real data $\mathbb{n}^t$는 $D^t$에 의해 1이란 값이 나와야함
- $F(E_C(\mathbb{m}^s)\;\vert\;E_S(\mathbb{n}^t))= \tilde{m}^t$
- generate한 데이터 $\tilde{m}^t$가 1에 가까워지도록 학습

## 3. Regression Loss
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00014.jpeg" width="100%">
</p> 

- generator 학습 안정화를 위한 loss
- discriminator 중간에 추출한 feature 값 사용
- $\mathcal{M}_t$: $t$ style을 가진 motion subset
- 같은 style $t$를 지닌 real data subset 평균과 generated data의 차이가 작아지도록 함
- L1 loss

## 4. Joint Embedding Loss
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00015.jpeg" width="100%">
</p> 

- style motion representation encoder에서 2D & 3D embedding이 같은 feature vector로 mapping되도록 하는 loss
- 우측: 2D와 3D embedding들이 같은 style label에 clustering이 잘 된 것 확인 가능

## 5. Style Triplet Loss
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00016.jpeg" width="100%">
</p>

- style clustering 성능 향상이 목적
- 같은 style은 가깝게, 다른 style은 margin ($\delta$)보다 더 멀리 있도록 제한
<br><br>

# Result
<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00017.jpeg" width="100%">
</p>

- 1st row: content
- 2nd row: style
- 3rd row: output
- style에 2D 혹은 3D 데이터를 주더라도 motion style transfer 결과가 똑같이 좋음
<br><br>

<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00018.jpeg" width="100%">
</p>

- (a) Xia, (b) Own dataset
- 빨간 상자로 표시된 old와 Heavy는 학습되지 않은 style (unseen data style)
- 학습되지 않았지만 다른 styler과 구별되고, 같은 style끼리 clustering 잘 됨(노란색 원) 
<br><br>

<p align="center">
<img src="./images/Unpaired_Motion_StyleTransfer/image_00019.jpeg" width="100%">
</p>

- 두 개의 style code를 linear하게 interpolate해서 사용 가능
- (a) depressed to proud
- (b) neutral to old
