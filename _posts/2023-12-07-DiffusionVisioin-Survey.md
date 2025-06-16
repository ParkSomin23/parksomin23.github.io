---
title: "Diffusion Models in Vision: A Survey"
date: 2023-12-07 15:00
category: "논문-리뷰"
tag: [Vision, Diffusion]
published: true

toc: true
toc_sticky: true
use_math: true
comments: true
---

> 논문: [Diffusion Models in Vision: A Survey](https://arxiv.org/abs/2209.04747v5)<br>
> 앞에서의 diffusion 발전 과정만 정리하였습니다.

# 1. Diffusion model 개요
- deep generative model의 종류로 2개의 stage으로 구성됨
    1. Forward Diffusion stage<br>
        <span style='background-color:#fff5b1'>Adding noise</span>, 단계적으로 Gaussian noise를 더하여 서서히 input 이미지를 왜곡시킴(perturbed)
    2. Reverse (Backward) diffusion stage<br>
        <span style='background-color:#fff5b1'>Denoising</span>, diffusion 과정을 역순으로 하는 단계로, diffused (noisy) data를 사용하여 원본 input 이미지를 복원하는 단계로, generative model을 학습함
- diffusion model 종류
    1. **denoising diffusion probabilistic models (DDPMs)**
        - non-equilibrium thermodynamics 이론이 바탕
        - latent variables를 사용하여 확률 분포를 추정하는 latent variable models
        - variational auto-encoders (VAEs)의 특별한 한 종류로 생각할 수 있음

            |  VAEs | DDPMs |
            | :---: | :---: |
            | encoding | forward diffusion |
            | decoding | reverse diffusion |

    2. **noise conditioned score networks (NCSNs)**
        - 다양한 noise level에서의 왜곡된(perturbed) 데이터의 분포에 대한 score function을 추정하는 neural network 학습, 이는 score matching 방법으로 구함
        - score function 정의는 the gradient of the log density
    3. **Stochastic differential equations (SDEs)**
        - DDPMs과 NCSNs의 일반화된 방법

# 2. Framework 설명
<p align="center">
    <img src="../assets/images/DiffusionVision-Survey/img_01.jpeg" width="80%">
</p>

## 2.1. Framework
- diffusion model은 probabilistic generative model의 한 종류
- 학습 데이터를 점진적으로 손상시키는(degrage) 과정에 대한 반대 과정을 학습함<br>
    즉, 손상된 이미지를 원래 이미지로 복원하는 방법을 학습
- 학습 과정 시에 2가지 process 사용: forward diffusion process, backward denoising process
    1. **forward diffusion process**
        - 학습 데이터에 noise를 더해가여 최종적으로 순수한 Gaussian noise를 만드는 과정 
        - 이 과정은 소량의 noise를 몇 단계에 거쳐 더하며, 각 단계에서의 noise의 크기는 달라짐
    2. **backward denoising process**
        - forward diffusion process를 단계에 거쳐 반대로 하는 과정
        - noise를 순서대로 제거하며 원래 이미지를 다시 만드는 과정으로, <span style='background-color:#fff5b1'>neural network를 학습시켜 각 단계에서 제거할 noise를 추정</span>
        - 차원 보존을 위해 U-Net 구조를 많이 사용
    3. **inference**
        - random white noise를 backward denoising process의 input으로 사용

## 2.2. DDPMs
denoising diffusion probabilistic models
1. **forward diffusion process**
    - $p(x_0)$: original data(index 0)의 data density, $\quad x_0 \sim p(x_0)$: uncorrupted training sample
    - $x_1, x_2, \cdots, x_T$: 아래 Markovian 과정에 의해 만들어진 noised version들

        $$p(x_t|x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\cdot x_{t-1},\ \beta_t\cdot I), \forall t \in \{1, \cdots, T\}$$

        - $T$: diffusion steps
        - $\beta_1, \cdots, \beta_T \in [0,\ 1]$: hyperparameters for variance schedule across diffusion steps 
        - $I$: input 이미지 $x_0$와 같은 차원의 identity matrix
        - $\mathcal{N}(x; \mu, \sigma)$: $x$를 생성하는 평균 $\mu$와 공분산 $\sigma$의 정규 분포
    - 위 수식이 재귀적이기에 균일 분포 (i.e. $\forall t \sim \mathcal{U}(\{1, \cdots, T\})$) 에서 $t$를 선택하면 $x_t$를 바로 구할 수 있음 (direct sampling)

        $$p(x_t|x_0) = \mathcal{N}(x_t;\ \sqrt{\hat{\beta_t}}\cdot x_0,\ (1-\hat{\beta_t})\cdot I)$$

        $$\alpha_t = 1 - \beta_t \quad \hat\beta_t=\Pi_{i=1}^t \alpha_i$$

        **variance schedule $\beta_t$를 고정하고, 원본 이미지 $x_0$를 알면 $x_t$를 바로 구할 수 있음**
    - backpropagation을 하기 위해, $p(x_t \vert x_0)$에서 뽑은 (sampled) $x_t$는 **reparametrization trick**에 의해 수식을 아래로 바꿔서 표현

        $$x_t = \sqrt{\hat\beta_t} \cdot x_0 + \sqrt{(1-\hat\beta_t)}\cdot z_t$$

        $$z_t \sim \mathcal{N}(0,I)$$
        
        - 정규화(standarize)의 역과정으로 Gaussian noise $z$에 표준 편차 ($\sqrt{(1-\hat\beta_t)}$)를 곱하고 평균 ($\sqrt{\beta_t} \cdot x_0$)을 더해줌
    - $\beta_t$ 특징
        - $x_T$의 분포가 표준 정규 분포 (Gaussian distribution) $\pi(x_T)=\mathcal{N}(0, I)$가 되어야 함
        - $p(x_T\vert x_0) = \mathcal{N}(x_T;\ \sqrt{\hat{\beta_T}}\cdot x_0,\ (1-\hat{\beta_T})\cdot I) = \pi(x_T)$가 성립되기 위해서, $\hat\beta_T\rightarrow 0$인 variance schedule $(\beta_t)^T_{t=1}$를 선택해야함
        - $(\beta_t)^T_{t=1} \ll 1$이면, reverse step은 forward step와 동일한 함수 형태(functional form)로 표현할 수 있음
            - $x_t$가 아주 작은 step에 의해 생성되었다는 가정이 있으면, $x_{t-1}$이 $x_t$와 가까운 영역에서 있었을 가능성이 매우 크기에, 이 영역을 Gaussian 분산으로 model하는 것이 가능 
        - [Ho et al.](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) 논문에서 사용한 variance schedule $(\beta_t)^T_{t=1}$: linearly increaing constants with $\beta_1=10^{-4}, \quad \beta_T=2\cdot 10^{-2}, \quad T = 1000$

2. **backward denoising process**
    - sample $x_T = \mathcal{N}(0, I)$를 시작으로 아래 수식처럼 거꾸러 가면 $p(x_0)$에서 새로운 sample들을 만들 수 있음

        $$p(x_{t-1}\vert x_t) = \mathcal{N}(x_{t-1};\ \mu(x_t, t),\ \Sigma(x_t, t))$$

    - neural network를 학습하여, 위 수식과 유사한 과정을 만드는 것이 목표

        $$p_\theta(x_{t-1}\vert x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t))$$

        - input: noisy image $x_t$ & embedding at time step $t$
        - <span style='background-color:#fff5b1'>learns</span>: 평균 $\mu_\theta(x_t, t)$ & 공분산 $\Sigma_\theta(x_t, t)$
    - model $p_\theta(x_0)$이 각 training sample $x_0$에 할당한 확률을 최대화하는 최대 우도 (maximum likelihood) 사용하는 것이 이상적이나, **$p_\theta(x_0)$를 구하기는 매우 어려움**<br>
    $\Rightarrow$ 이를 해결하기 위해 <span style='background-color:#fff5b1'>negative log likelihood의 variational lower bound / ELBO (Evidence Lower BOund)를 최소화</span>하는 방법 사용

        $$\mathcal{L}_{vlb} = -log\ p_\theta(x_0\vert x_1) + KL(p(x_T\vert x_0)\Vert\pi(x_T)) + \sum_{t>1}KL(p(x_{t-1}\vert x_t, x_0)\ \Vert\ p_\theta(x_{t-1}\vert x_t))$$
        
        - KL: 두 확률 분포의 Kullback-Leibler divergence
        - 두 번째 항은 $\theta$에 영향을 받지 않기에 무시 가능
        - 마지막 항은 **각 time step $t$에서 $p_\theta(x_{t-1}\vert x_t)$가 forward process가 원본 이미지를 조건으로 받을 때의 true posterior에 최대한 가까워지도록 neural network가 학습 됨**
        - KL divergence의 closed-form expression에 의해 $p(x_{t-1}\vert x_t, x_0)$이 Gaussian distribution임을 증명할 수 있음
    - <details>
        <summary>variational bound 증명: Appendix A</summary>
        <div markdown="1">

        - VAEs에서 사용한 방법과 비슷

            | VAEs  | Diffusion |
            | :---: | :---: |
            |latent variables  | noisy images $x_{1:T}$ |
            |observed variable | original image $x_0$   |

            <br>

            $$\begin{align}
                log\ p_\theta(x_0) &= log\int p_\theta(x_{0:T})\ \partial x_{1:T} \\ 
                &= log\int p_\theta(x_{0:T})\cdot\frac{p(x_{1:T} | x_0)}{p_(x_{1:T} | x_0)} \partial x_{1:T} \\ 
                &= log\int p(x_{1:T}|x_0)\cdot\frac{p_\theta(x_{0:T})}{p(x_{1:T} | x_0)} \partial x_{1:T} \\
                &= log\ \mathbb{E}_{x_{1:T}\sim p(x_{1:T}|x_0)} [ \frac{p_\theta(x_{0:T})}{p(x_{1:T} | x_0)} ] 
            \end{align}$$
            
            - (1): $p_\theta(x_0)$에 의한 정의
            - (3): $x_{1:T}$에 의한 편미분 수식을 만들기 위해 위치 바꿈
            - (4): $\mathbb{E}$으로 정리
            <br><br>
            - Jensen's inequality에 의해 random variable $Y$와 convex function $f$는 아래 부등식이 성립함

                $$f(\mathbb{E}[Y]) \leq \mathbb{E}[f(Y)]$$
                
                $f$는 $log$, $Y$는 $\frac{p_\theta(x_{0:T})}{p(x_{1:T} \vert x_0)}$로 $log$ 함수는 concave하기에 위 부등식을 바꿔서 정리하면,
                
                $$log\ p_\theta(x_0) \geq \mathbb{E}_{x_{1:T}\sim p(x_{1:T}|x_0)}\ [\ log\frac{p_\theta(x_{0:T})}{p(x_{1:T} | x_0)}\ ]\\
                -log\ p_\theta(x_0) \leq \mathbb{E}_{x_{1:T}\sim p(x_{1:T}|x_0)}\ [\ log \frac{p(x_{1:T} | x_0)}{p_\theta(x_{0:T})}\ ]$$

                구하기 힘든 $p_\theta(x_0)$이 아닌 부등식의 오른쪽 항을 최소화시키는 걸 objective function으로 사용 가능해짐

            - 정의에 의해 forward와 reverse process는 Markovian으로, 확률들을 아래와 같이 다시 정의할 수 있음
                <p align="center">
                    <img src="../assets/images/DiffusionVision-Survey/img_02.jpeg" width="60%">
                </p>
            <br>

            - 위에서 정리된 확률로 부등식의 오른쪽 항을 정하면, 
                $$\begin{align}
                \mathbb{E}_{x_{1:T}\sim p(x_{1:T}|x_0)}\ [\ log \frac{p(x_{1:T} | x_0)}{p_\theta(x_{0:T})}\ ] 
                &= \mathbb{E}_p [\ log \frac{\Pi_{t=1}^T\ p(x_t | x_{t-1})} {p_\theta(x_T)\ \Pi_{t=1}^T\ p_\theta(x_{t-1} | x_{t})}] \\
                &= \mathbb{E}_p[\ -log\ p_\theta(x_T) + \sum_{t=1}^Tlog\frac{p(x_t | x_{t-1}) }{p_\theta(x_{t-1} | x_{t})}] \\
                &= \mathbb{E}_p[\ -log\ p_\theta(x_T) + \sum_{t=1}^Tlog\frac{p(x_{t-1}|x_t, x_0)\cdot p(x_t|x_0)} {p(x_{t-1}|x_0)\cdot p_\theta(x_{t-1} | x_{t})}] \\
                &= \mathbb{E}_p[\ -log\ p_\theta(x_T)] + \mathbb{E}_p[ \sum_{t=2}^T log\frac{p(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_0)}] \\ 
                &+ \mathbb{E}_p[ \sum_{t=2}^T log\frac{p(x_t|x_0)}{p(x_{t-1}|x_0)} + log\frac{p(x_1|x_0)}{p_\theta(x_0|x_1)}] \\
                &= \mathbb{E}_p[\ -log\ p_\theta(x_T)] + \mathbb{E}_p[ \sum_{t=2}^T log\frac{p(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_0)}] \\ 
                &+ \mathbb{E}_p[  log\frac{p(x_T|x_0)}{p(x_1|x_0)} + log\frac{p(x_1|x_0)}{p_\theta(x_0|x_1)}] \\
                &= \mathbb{E}_p[log\frac{1}{p_\theta(x_T)}\cdot \frac{p(x_T|x_0)}{p(x_1|x_0)} \cdot \frac{p(x_1|x_0)}{p_\theta(x_0|x_1)}]  + \mathbb{E}_p[ \sum_{t=2}^T log\frac{p(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_0)}] \\ 
                &= \mathbb{E}_p[log\frac{p(x_T|x_0)}{p_\theta(x_T)} - log\ p_\theta(x_0|x_1)] + \mathbb{E}_p[ \sum_{t=2}^T log\frac{p(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_0)}] \\
                &= KL(p(x_T|x_0)\ \Vert\ p_\theta(x_T)) - log\ p_\theta(x_0|x_1) \\
                &+\sum_{t=2}^T KL(p(x_{t-1}|x_t, x_0)\ \Vert\ p_\theta(x_{t-1}|x_t)))
                \end{align}$$

                - (5): 편의상 $x_{1:T}\sim p(x_{1:T}\vert x_0)$를 $p$로 대체
                - (7): forward process가 Markovian이기에 $p(x_t\vert x_{t-1}) = p(x_t\vert x_{t-1}, x_0)$이며, 베이지언 정리에 의해 아래의 수식이 성립됨
                    $$ p(x_t\vert x_{t-1}, x_0) = \frac{p(x_{t-1}\vert x_t, x_0)\cdot p(x_t\vert x_0)}{p(x_{t-1}\vert x_0)}$$
                - (8) & (9): $t \geq 2$에 대해서 정리하고, (9)의 마지막 항은 (7)에서 t=1일 때 나오는 값
                - (11): (9)의 첫 번째 항은 전개되며 정리됨
                - (14) & (15): Kullback-Leibler divergence로 바꾸기

        </div>
        </details>

    - [Ho et al.](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) 논문에서 공분산 $\Sigma_\theta(x_t, t)$를 상수로 정의하고, 평균 $\mu_\theta(x_t,t)$를 noise에 대한 함수로 표현하는 방법 제안

        - $$\mu_\theta=\frac{1}{\sqrt\alpha_t}\cdot(x_t - \frac{1-\alpha_t}{\sqrt{1-\hat\beta_t}}\cdot z_\theta(x_t,t))$$

        - 위 수식을 기반으로 $\mathcal{L}_{vlb}$를 random한 time step $t$의 forward process에서의 예측된 noise $z{\theta}(x_t,t)$ 와 실제 noise $z_t$ 사이의 거리 비교로 식을 간단하게 변환    

            $$\mathcal{L}_{simple} = \mathbb{E}_{t\sim[1,T]} \mathbb{E}_{x_0\sim p(x_0)} \mathbb{E}_{z_t\sim \mathcal{N}(0, I)} \Vert z_t - z_\theta(x_t, t)\Vert^2$$

            - $z_\theta(x_t, t)$: network predicting the noise in $x_t$
            - $x_t$: sampled via $x_t = \sqrt{\hat\beta_t} \cdot x_0 + \sqrt{(1-\hat\beta_t)}\cdot z_t$, where we use a random image $x_0$ from the training set
        - generative process는 $p_\theta(x_{t-1}\vert x_t)$에 의해 정의되지만, neural network가 평균과 공분산을 바로 추측하는 것이 아닌,<br> 
        <span style='background-color:#fff5b1'>**image에서의 noise를 예측 $\rightarrow$ 평균은 $\mu_\theta$에 대한 수식으로 구하고, 공분산은 고정된 상수이므로 그대로 사용**</span>
        - 전체 과정에 대한 알고리즘
            <p align="center">
                <img src="../assets/images/DiffusionVision-Survey/img_03.jpeg" width="60%">
            </p>
        - <details>
            <summary>수식 유도: Appendix B</summary>
            <div markdown="1">

            - $p_\theta(x_{t-1}\vert x_t)$의 공분산을 미리 $\sigma_t^2\cdot I$로 고정하여, 학습하지 않도록 제한
            - <span style='background-color:#fff5b1'>**$\sigma_t^2=\beta_t$로 고정되므로, $\mathcal{L}_{vlb}$의 Kullback-Leibler divergence가 두 분포의 평균 사이의 거리와 $\theta$에 영향 받지 않는 상수의 합으로 정리됨**</span>

                $$\begin{align*}
                \mathcal{L}_{kl} &= KL(p(x_{t-1}|x_t, x_0)\ \Vert\ p_\theta(x_{t-1}|x_t))\\
                                    &= \frac{1}{2\cdot\sigma_t^2}\cdot \Vert \tilde{\mu}(x_t, x_0) - \mu_\theta(x_y,t)\Vert^2 + C
                \end{align*}$$

                - $\tilde{\mu}(x_t, x_0)$: $p(x_{t-1}\vert x_t, x_0)$의 평균 $\qquad \mu_\theta(x_y,t)$: $p_\theta(x_{t-1}\vert x_t)$의 평균 $\qquad C$: 상수
                - neural network의 output은 $\mu_\theta(x_y,t)$
            - 평균 $\tilde{\mu}(x_t, x_0)$를 $x_t$와 $z_t$으로 표현하여 정리가 가능해지며, $\mu_\theta(x_t, t)$ 또한 이와 가까워져야 함

                $$\tilde{\mu}(x_t, x_0) = \frac{1}{\sqrt\alpha_t}(x_t - \frac{\beta_t}{\sqrt{1-\hat\beta_t}}\cdot z_t)$$

                $$\mu(x_t, t) = \frac{1}{\sqrt\alpha_t}(x_t - \frac{\beta_t}{\sqrt{1-\hat\beta_t}}\cdot z_\theta(x_t, t))$$

                $z_\theta(x_t, t)$: neural network output, noisy image $x_t$가 주어졌을 때 noise $z_t$ 추측값
            - $\mathcal{L}_{kl}$의 위 수식에서의 평균 값으로 대체하면 아래 수식으로 정리됨

                $$\mathcal{L}_{kl}=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\hat\beta_t)}\Vert z_t - z_\theta(x_t, t)\Vert^2$$

                이미지 $x_t$의 실제 noise와 network가 예측한 값 사이의 시간에 따른(time-weighted) 거리 의미
            - 앞의 weight인 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\hat\beta_t)}$를 생략해서 더 간단하게 만들어 최종 loss 수식 전개
            
                $$\mathcal{L}_{simple}=\mathbb{E}_{t\sim[1,T]} \mathbb{E}_{x_0\sim p(x_0)} \mathbb{E}_{z_t\sim \mathcal{N}(0, I)} \Vert z_t - z_\theta(x_t, t)\Vert^2$$

            </div>
            </details>

## 2.3. NCSN
- Noise Conditioned Score Network
- $\nabla_x log\ p(x)$
    - 몇몇의 data density $p(x)$의 score function은 input에 대한 log density의 gradient으로 정의 가능
    - gradient의 방향성은 random sample ($x_0$)를 밀도가 높은 영역에 있는 samples ($x_N$)로 옮기는 Langevin dynamics algorithm에서 사용됨
    - Langevin dynamics는 data sampling애 사용할 수 있는 반복적인 방법
    - 물리학과의 비교
        물리학에서는 입자와 다른 분자들 사이의 상호작용을 고려한 분자 시스템에서 입자의 궤적 결정을 위한 방법으로 drag force와 random force에 영향 받음

        | difference | drag force | random force $\omega_i$ | 두 force에 대한 weight $\gamma$ |
        | :---:     | :---: | :---: | :---: |
        | physics   |  시스템 안의 항력 (drag force)| 분자 사이의 빠른 상호작용으로 인해 만들어진 random force |입자가 존재하는 공간애서 환경의 마찰 계수(friction coefficient) |
        | diffusion | log density의 gradient로 data space에서의 random sample을 밀도 놓은 data density $p(x)$로 끌어들이는 힘  | local minima에서 벗어나게 해주는 요소 | update에서의 magnitude 정도 조절 |

        - iterative updates of the Langevin dynamics
            
            $$x_i = x_{i-1} + \frac{\gamma}{2} \nabla_x log\ p(x)+\sqrt\gamma\cdot \omega_i$$

            - $i \in \{1, \cdots, N\},\ \text{recursively for}\ N\rightarrow \infty \ \text{steps}$
            - $\gamma$: score 방향성으로의 update magnitude 조절
            - $x_0$: prior distribution에서 sample 됨
            - $\omega_i\sim \mathcal{N}(0,I)$: local minima에서 나올 수 있게 도와주는 noise  
    - neural network $s_\theta(x) \approx \nabla_x log\ p(x)$로 score를 예측한 후, p(x)에서 sampling하는 방법으로 generative model에 적용 가능
        - score mathching 방법으로 학습가능하지만, $\nabla_x log\ p(x)$를 모르기에 아래 수식을 그대로 적용할 수 없음<br>
            denoising score matching이나 sliced score mathcing 방법을 사용해야 함

            $$\mathcal{L}_{sm} = \mathbb{E}_{x\sim p(x)}\Vert s_\theta(x) - \nabla_x log\ p(x)\Vert^2_2$$

    - 실제 데이터에서 적용할 때 manifold hypothesis에 관련된 문제들이 발생함: 
        데이터가 low-dimensional manifold에 있을 때, score estimation $s_\theta(x)$가 일관되지 않음<br>
        이로 인해 밀도가 높은 지역으로 Langevin dynamics가 수렴하지 않을 수 있게 됨
    - 이를 해결하기 위해, 데이터를 **다양한 scale의 Gaussian noise**애 대해 왜곡(perturbing)하고, 하나의 NCSN를 학습하여 noisy 분포에 대한 score estimate 진행<br>
    각 noise scale에 대한 score estimates 사용
        - $\sigma_1 < \sigma_2 < \cdots < \sigma_T$: a sequence of Gaussian noise scales such that $p_{\sigma_1}(x) \approx p(x_0)$ and $p_{\sigma_T}\approx\mathcal{N}(0,I)$
        - $s_\theta(x, \sigma_t) \approx \nabla_x log\ p_{\theta_t}(x)$를 달성하기 위해 NCSN $s_\theta(x, \sigma_t)$를 denoising score matching으로 학습 $(\forall t \in \{1, \cdots, T\})$

            $$\begin{align*}
            p_{\sigma_t}(x_t|x) &= \mathcal{N}(x_t;\ x,\ \sigma_t^2\cdot I)\\
                &= \frac{1}{\sigma_t\cdot \sqrt{2\pi}}\cdot exp(-\frac{1}{2}\cdot(\frac{x_t-x}{\sigma_t})^2)
            \end{align*}$$ 

            일 때, $\nabla_{x} log\ p_{\sigma_t}(x)$를 아래의 수식처럼 유도 가능 ($x_t$: $x$의 noised version)
            
            $$\nabla_{x_t} log\ p_{\sigma_t}(x_t|x) = -\frac{x_t-x}{\sigma_t^2}$$

            모든 $(\sigma_{t})^T_{t=1}$에 대해 일반화하고, gradient를 $\mathcal{L}_{sm}$ 대입하면 아래처럼 간단하게 정리됨 $(\forall t\in\{1, \cdots, T\})$

            $$\mathcal{L}_{dsm}=\frac{1}{T}\sum_{t=1}^T\lambda(\sigma_t) \mathbb{E}_{p(x)} \mathbb{E}_{x_t \sim p_{\sigma_t}(x_t|x)}\Vert s_\theta(x_t,\sigma_t)+\frac{x_t-x}{\sigma_t^2} \Vert^2_2$$

            $\lambda(\sigma_t)$: weighting function

            학습이 완료된 후, neural network $s_\theta(x_t, \sigma_t)$는 time step $t$에서 noisy input $x_t$에 대한 score $\nabla_{x_t} log\ p_{\sigma_t}(x_t)$에 대한 추측값을 return하게 됨<br>
    - Inference 시에 annealed Langevin dynamics 사용
        <p align="center">
        <img src="../assets/images/DiffusionVision-Survey/img_04.jpeg" width="60%">
        </p>
        
## 2.4. SDE
- Stochastic Differential Equations
- data distribution $p(x_0)$을 서서히 noise로 바꾸는 방법으로, **위의 2가지 방법을 일반화함**<br>
diffusion 과정이 **연속적 (continuous)**으로 고려되어, stochastic differential equation (SDE)의 해가 되기 때문<br><br>
이 방법의 diffusion의 역과정은 reverse-time SDE로 구할 수 있는데, 각 time step에서의 밀도에 대한 score function이 필요함<br><br>
이를 위해 neural network는 score function들을 예측하고, numerical SDE solvers를 사용하여 $p(x_0)$에서의 sample들을 생성하는 방식 제안됨<br><br>
즉, NCSNs 방법처럼 왜곡된 data와 time step을 입력 받고, score function의 예측값 생성

1. **forward diffusion process** $(x_t)_{t=0}^T, t\in [0, T]$
    
    $$\frac{\partial x}{\partial t}=f(x,t)+\sigma(t)\cdot \omega_t \Leftrightarrow \partial x=f(x, t)\cdot \partial t + \sigma(t)\cdot\partial \omega$$

    - $\omega_t$: Gaussian noise
    - $f$: drift coefficient 연산하는 함수
    - $\sigma$: 시간에 따라, diffusion coefficient 연산하는 함수
    
    diffusion이 SDE의 해가 되기 위해서,<br>
    1\) drift coefficient은 점진적으로 data $x_0$를 무효되게 (nullify) 디자인<br>
    2\) diffusion coefficient는 더해질 Gaussian noise 조절

2. **reverse-time SDE**

    $$\partial x = [f(x,t) - \sigma(t)^2\cdot \nabla_xlog\ p_t(x)]\cdot \partial t + \sigma(t) \cdot \partial \hat\omega$$

    - $\hat\omega$: 시간이 T에서 0으로 거꾸로 뒤집혔을 때의 Brownian motion

    순수한 noise에서 시작하면, data destruction을 한 drift를 제거함으로써 data를 복원할 수 있음을 나타냄<br>
    즉, $\sigma(t)^2\cdot \nabla_xlog\ p_t(x)$를 빼줌으로써 drift 제거 가능

    neural network $s_\theta(x, t) \approx \nabla_xlog\ p_t(x)$를, NCSNs에서의 objective에서 연속적인 경우를 적용하여 사용하면 됨

    $$\mathcal{L}_{dsm}^*=\mathbb{E}_t [ \lambda(t)\ \mathbb{E}_{p(x_0)}\ \mathbb{E}_{p_t(x_t|x_0)}\ \Vert s_\theta(x_t, t) - \nabla_{x_t}log\ p_t(x_t|x_0) \Vert^2_2 ]$$

    - $\lambda$: weighting function $\qquad t \sim \mathcal{U}([0,T])$
    
    drift coefficient $f$는 affine하면, $p_t(x_t\vert x_0)$는 Gaussian 분포를 따름<br>
    $f$가 affine이지 않으면, denoising score matching 사용 불가하며 sliced score matching로 대체해야 함(fallback)

    **reverse-time SDE**에서 첫 번째 수식으로 정의된 SDE에 모든 numerical 방법으로 sampling이 가능하지만,<br>
    실제 solver들은 연속적으로 작동하지 않기에 다른 방법을 써야함<br>
    1. Euler-Maruyama method
        - 작은 negative step $\Delta t$로 고정하고, 처음 time step $t=T$가 $t=0$이 될 때까지 Algorithm 3를 반복
            <p align="center">
            <img src="../assets/images/DiffusionVision-Survey/img_05.jpeg" width="60%">
            </p>
        - Brownian motion: $\Delta\hat\omega=\sqrt{\vert \Delta t\vert}\cdot z,\quad z\sim\mathcal(0,I)$
    2. Predictor-Corrector sampler
        - 더 나은 example 생성하도록하는 sampling 방법
        - reverse-time SDE에서 sample하는 numerical 방법 사용하고, corrector로 score-based 방식 사용 (ex) (이전 subsection에 있는) annealed Langevin dynamics
        - reverse process를 model할 때, ordinary differential equations (ODEs)도 사용 가능<br>
        따라서, SDE 해석으로 나온 새로운 sampling 방법은 ODEs에 적용된 numerical 방법을 기반으로 함<br>
        효율성이 좋다는 장점 있음

# 3. Relation to Other Generative Models
## 3.1. VAEs
- 공통점
    - data가 latent space에서 mapping 됨
    - latent representations를 데이터로 바꿔주는 생성하는 과정을 학습함
    - objective function은 lower-bound of the data likelihood에서 유래됨
- 차이점

    | | latent representation | dimension size | mapping to the latent space |
    | :---: | :---: | :---: | :---: |
    |VAEs| 원본 이미지의 압축된 정보를 담고 있음 | 입력 데이터보다 차원이 줄어들 때 더 잘 작동됨 | 학습 가능 |
    |Diffusion | forward process의 마지막 step 이후에는 data를 완전히 파괴함 | 원본 데이터와 차원 크기가 같음 | forward process는 학습 불가능 (원본 이미지에 Gaussian noise를 점진적으로 더하면서 latent를 구하기 때문) |

## 3.2. Autoregressive models
- Autoregressive model들은 이미지를 pixel들의 순서로 나타냄<br>
    전에 생성한 pixel을 조건으로 한, pixel by pixel로 이미지 생성해서 새로운 sample 생성<br>
    $\Rightarrow$ 단방향적 경향(unidirectional bias)이라는 한계 존재
- [Esser et al.](https://proceedings.neurips.cc/paper/2021/file/1cdf14d1e3699d61d237cf76ce1c2dca-Paper.pdf)에서 Autoregressive models과 diffusion model은 서로 상호보완적이며 위의 문제를 해결할 수 있다고 함<br>
 각 transition이 autoregressive model로 구현한 Markov chain로, multinomial diffusion process의 역 과정을 학습하는 방식 사용<br>
 Markov chain에서 이전 step이 autoregressive model에 global information 제공

## 3.3. Normalizing flows
- Normalizing flows는 간단한 Gaussian 분포를 복잡한 데이터 분포로 변환하는 방법으로, 변환은 계산하기 쉬운 Jacobian determinant을 가진 invertable(뒤집을 수 있는) 함수의 집합에 의해 수행됨
- **likelihood가 추적 가능함** $\Rightarrow$ objective function은 negative log-likelihood 학습
- 공통점:
    - 데이터 분포를 Gaussian noise로 mapping
- 차이점:
    - invertable하고 미분가능한 함수들을 활용하여 학습하기에 mapping이 결정됨 (deterministic fashion)<br>
    즉, network 구조와 forward process에 대해 diffusion보다 추가적인 제한 조건이 있음
- 두 방식을 결함한 방법이 DiffFlow<br>
    forward와 reverse process들이 둘 다 학습 가능하고 확률론적임(stochastic)

## 3.4. Energy-based models (EBMs)
- energy function (정규화 되지 않은 density function의 추정치)를 제공하는데에 집중<br>
    $\Rightarrow$ likelihood 기반 방식과 대조적으로, regression neural network 사용 가능<br>
    단점은 flexibility가 크기에 학습이 어려움
- 제안되는 학습 방법으로 score matching을 사용하고, sampling에서는 score function에 기반으로 하는 Markov Chain Monte Carlo (MCMC) 방법을 많이 사용<br>
$\Rightarrow$ NCSNs은 학습과 sampling이 score function만 필요로 하는 energy-based framework의 한 방식

## 3.5. GANs

| | 단점 | 장점 | latent space | 의미론적인 (semantic) 성질 | 
| :---: | :---: | :---: | :---: | :---: |
|GANs| adversarial objective 때문에 학습이 어렵고 종종 mode collapse 발생 | efficient | low-dimensional latent space | subspace들이 시각적인 특성 나타내어, latent space를 바꾸면서 특성 조작 가능 | 
|Diffusion | inefficient (inference 시에 여러 network evaluation 필요) | likelihood 기반이기에 학습 과정이 안정되고 더 다양성을 보임 | 이미지의 dimension 크기 유지, random Gaussian distribution으로 나타남 | guidance technique를 사용하는데, latent space에서 semantic 특성을 나타내지 않음 |

- [Song et al.](https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf)이 diffusion model의 latent space는 정의가 명확한 구조(well-defined structure)를 가지고 있으며, 이 공간에서 interpolations하면 이미지 공간에서 interpolation 된다고 설명함<br>
즉, diffusion의 latent space에 대한 연구가 GAN보다 덜 되었으며 후속 연구들이 필요함을 의미함

# 4. 개인적인 정리
<p align="center">
    <img src="../assets/images/DiffusionVision-Survey/Diffusion_01.jpeg" width="80%">
    <img src="../assets/images/DiffusionVision-Survey/Diffusion_02.jpeg" width="80%">
    <img src="../assets/images/DiffusionVision-Survey/Diffusion_03.jpeg" width="80%">
</p>