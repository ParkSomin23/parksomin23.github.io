---
#layout: post
title: "Self-Supervised Learning 개요"
#subtitle: "[Tips]"
date: 2022-05-25 14:22
category: "논문-리뷰" 
tag: [Self-Supervised Learning]
#classes: wide
published: true

toc: true
toc_sticky: true
---

# 1. What is Self-Supervised Learning?

| Learnings  | explaination |
| --- | --- |
| Supervised Learning  | Learn with **“labeled”** data |
| Weakly Supervised Learning | Learn with **"coarse"** labels ex) segmentation with scribbles |
| Semi-Supervised Learning | Learn with **“labeled + unlabeled”** data ex) MIL(Multiple Instance Learning) |
| Self-Supervised Learning | Learn with **“unlabeled”** data |

[Image Feature Learning - supervised, self-supervised, semi-supervised, weakly-supervised](https://nuguziii.github.io/survey/S-004/)  
 <br/>

# 2. 2020 Samsung AI Forum - Yann LeCun

[2020 Samsung AI Forum Youtube Link](https://youtu.be/BqgnnrojVBI)


> **사람과 동물들은 supervised하게 학습하지 않는다! 추론(reasoning)하는 model이 필요하다  
  Capture Dependencies. Predict everything from everything else.**
>

1. Learn Hierarchical Representations of the world   
2. Learn Predictive(Forward) models of the world   
  
> **Question**: How to represent uncertainty/multi modality in the prediction?  
  **‼️**: Energy-Based Functions (연속적인 데이터는 low energy, 불연속적인 데이터는 high energy)
>

## **1. Contrastive/Non-Contrastive Learning**
1. **Masked Auto-Encoder**
    - BERT/RoBERTa :  corrupt data and tries to reconstruct the text (+Transformer)
    - 미적분까지 하는 모델 있음
    - MMBlenderbot: 사진보고 chatting 가능
    - DERT : Conv(frontend) + Transformer(backend)
    - 하지만 reconstruction model을 subsequent image recognition system에서 별로 안 좋았은 결과가 나왔음  
        <br/>

2.  **Contrastive Embedding**
    - **Audio에서 Wav2Vec2**
    - 동일한 encoder를 사용하고 positive sample(원본에서 augumentation)의 energy는 크게, negative sample의 energy는 크게 만듦
    - 단점은 데이터 많아야하며, 어떤 sample을 negative으로 할지도 골라야함(hard-negative mining을 사용)  
    그래도 잘 작동함(특징 추출 괜찮음) 
    - maximum likelihood의 일종
    - NCE(Noise Contrastive Estimation): batch >> softmax >> negative push large, positive pull small      
    <br/>

3. **Non-Contrastive Embedding**
    - BYOL - Bootstrap Your Own Latent
    - 같은 encoder가 아닌 weight 값을 조금 다르게 해준 encoder를 가지고 학습을 진행
    - negative sampling 필요가 없음
    - 하지만 왜 잘 작동하는 이유를 모름
    - DeepCluster, SwAV도 있음(다른 방식)  
        <br/>
         
## **2. Regularized/Architectural Learning**
- limit the volume of y space that can take low energy
- 대표적인게 K-Means >> 하지만 high-dimension에서 별로 안 좋음
- latent variable model을 사용하는 이유는 multi-modal한 방식으로 예측이 가능
    - ex) 어떤 값을 바꾸면 그 값을 따라 바뀌는..
    - Sparse Encoding/Coding, VAE
        

[참고 블로그 : "Self-Supervised learning이란?"](https://89douner.tistory.com/332)