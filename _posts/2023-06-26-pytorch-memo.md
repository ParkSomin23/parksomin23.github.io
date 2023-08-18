---
#layout: posts
title: "PYTORCH Memos"
date: 2023-06-26 13:18
category: "정리함"
tag: ["pytorch"]
published: true
toc: true
toc_sticky: true
---

## **Tensor vs tensor**
- Tensor: class <br>
T = torch.Tensor() : 빈 tensor<br>
int to float<br>
torch data이면, 같은 메모리 공간<br>
list나 numpy면, copy 후 생성

- tensor: function<br>
T = torch.tensor() : 데이터가 없으므로 error<br>
int to int<br>
memory copy
- [[Pytorch] Tensor 기본 사용법](https://amber-chaeeunk.tistory.com/84)

## **nn vs F**
- nn : class
- F : 함수
- nn.CrossEntropy  : weight 한 번만 pass <br>
  F.cross_entropy : weight 게속 pass
- [[개발팁] torch.nn 과 torch.nn.functional 어느 것을 써야 하나?](https://cvml.tistory.com/10)
    
## **matmul**

- | dot | 벡터 X 벡터 | 내용 |
  |:---: | :---: | :---: |
  | mm | 행렬 X 행렬 | broadcasting 지원 안 됨 |
  | matmul | 행렬 X 행렬 , 행렬 X 벡터 | broadcasting 지원됨 |

## **view vs reshape**
    
permute과 view는 모양만 바뀌는 거지만, reshape는 rearrange

- view
copy하지 않고 같은 메모리 주소 사용
원래 tensor 값 바꾸면 같이 바뀜
    
    ```python
    a = torch.zeros((3,2))
    b = a.view(2,3)
    a.fill_(1)
    
    => a = [[1,1], [1,1], [1,1]]
    => b = [[1,1,1], [1,1,1]]
    ```
    
- reshape
copy 해서 reshape
원래 tensor와 구별됨
    
    ```python
    a = torch.zeros((3,2))
    b = a.reshape(2,3)
    a.fill_(1)
    
    => a = [[1,1], [1,1], [1,1]]
    => b = [[0,0,0], [0,0,0]]
    ```
        
## **nn.RNN Mask**

RNN에 dropout을 적용하려면 time step이 흘러도 [dropout mask가 바뀌지 않는 방식](https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b)으로 구현되어야만 올바르게 동작한다. <br>
그런데 [GRU나 LSTM에 구현된 dropout은 mask가 매 time step마다 바뀌게](https://discuss.pytorch.org/t/dropout-for-rnns/633/11) 구현되어 있다고 한다. <br> 
따라서 직접 mask를 fix시켜서 구현해주거나, [LockedDropout](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.nn.html)과 같은 모듈을 사용해야한다.
    
## **torch tracing & annotation(script)**
- **torch tracing**이란, 입력값을 사용하여 모델 구조를 파악한 뒤, 입력값의 흐름을 통해 모델 기록<br>
flow가 기록되기에 statically fix됨
- **torch script**이란, 컴파일러가 직접 모델 코드 분석하여 컴파일 진행하기에 Dynamic한 control(if, break) 사용가능<br>
하지만 지원하지 않음 python code 및 type 추정 문제 있어서 확인 필요
- [[PytorchToC++] 01. TorchScript 비교 및 simple code review](https://data-gardner.tistory.com/m/105)

## Faster Training
```python
for param in model.parameters():
    param.grad = None
```
- [Best Performance Tuning Practices for Pytorch](https://ai.plainenglish.io/best-performance-tuning-practices-for-pytorch-3ef06329d5fe)
   
## DataLoader
[pytorch dataset 정리](https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/)
    
## link
[Pytorch 개발 팁](https://newsight.tistory.com/301)
        
