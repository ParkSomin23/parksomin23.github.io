---
title: "Generative Agents: Interactive Simulacra of Human Behavior"
date: 2025-06-13 17:32
category: "논문-리뷰"
tag: [Generative Agents, LLM, Simulation]
published: true

toc: true
toc_sticky: true
use_math: true
---
> [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)<br>
> [[코드](https://github.com/joonspk-research/generative_agents)]


<figure class="centered-figure">
  <img src="../assets/images/Generative-Agents/img_0001.jpeg"  style="width: 80%;">
  <figcaption class="caption">Smallville 마을에서 상호작용하는 Generative Agents</figcaption>
</figure>



# 📌 이 논문을 읽은 이유
<!-- 줄글로 작성 (내 동기, 관심사, 맥락 설명)
예시: 
이 논문은 제가 진행 중인 [프로젝트/연구 분야]와 관련된 문제를 다루고 있어서 읽게 되었습니다. 
특히 기존 방법에서 느꼈던 [문제점]을 개선한 접근이라는 점에서 흥미를 가졌습니다.
-->
최근 Agent AI, MCP 등 LLM의 가능성과 활용성이 점점 더 넓어지고 있다. <br>
그러다 보니 언젠가는 꼭 읽어봐야지 생각했던 해당 논문이 자연스레 생각나서 읽게 되었다.<br><br>


# 💡 논문의 주요 내용 간단 요약
<!-- 줄글로 작성 (내 해석 중심)
예시: 
이 논문은 [문제 정의]를 해결하기 위해 [핵심 아이디어]를 제안합니다. 
기존 방법과 비교했을 때, [차별점/장점]이 두드러지며 [어떤 상황]에서 특히 효과적입니다.
-->
이 논문은 LLM(chatGPT-3.5 turbo)을 기반으로 generative agent를 만들었다. <br>
Generative agent 구조는 memory stream, reflection, planning 구조를 가진다. 각 구조는 아래와 같이 구성되어 있다:
- <b>Memory stream</b>: 사람의 뇌처럼 기억을 저장하고, 연관된 기억을 다시 불러오는 구조로, 이를 바탕으로 각 agent들이 다음 행동을 결정한다.
- <b>Reflection</b>: 저장된 기억을 모아 고차원의 생각으로 발젼시키는 구조이다.
- <b>Planning</b>: reflection의 결과와 현재 환경을 고려하여 고차원적인 계획을 세우는 구조이다. 해당 계획을 통해 행동이나 반응이 결정되고, 이게 다시 memory stream에 저장된다. 

즉, agent들이 기억하고 / 관련된 기억을 떠올리고 / 기억을 종합하여 의미를 도출해 내고 / 기억에 따라 반응과 행동을 하는데 / 이를 다시 기억에 반영하는 구조를 만들어서 사람처럼 행동하도록 한 논문이다. 

위의 구조는 자연어로 저장 및 사용되며 agent의 모든 행동과 생각은 LLM을 통해서 얻게 된다. <br>
(ex: "초기 agent 설정, 기억과 계획" + "다음에 이 캐릭터는 무슨 행동을 하게 될까?" $---$ LLM $\rightarrow$ "다음 행동")

실험은 25명의 agent들이 사는 Smallville 마을에서 상호작용하면서 진행되며, 평가자들은 agent들을 인터뷰하면서 generative agent의 가능성을 확인했다. <br><br>


# 📚 핵심 아이디어와 주요 기법 정리
<!-- Bullet 형태로 작성 (논문 내용 요약 / 정리)
예시:
- 논문은 [기법/구성요소1]을 통해 [문제 A]를 해결함.
- [기법/구성요소2]는 [어떤 점]에서 기존보다 나은 성능을 보임.
- [기법/구성요소3]는 추가적으로 [효과/장점]을 제공함.
-->
## 이전 방법의 장단점과 논문에서 제안한 내용

| 방법 | 장점 | 단점 |
| :--: |:--: |:--: |
|rule-based|과거부터 많이 사용됨 <br> 기초적인 상호작용이 가능함 | 새로운 상황에 대처하지 못함|
|learning-based| 시림보다 뛰어난 agent들이 나오고 있음 | 기존에 있는 게임과 같이 보상 체계가 정해진 게임에서만 달성할 수 있음 <br>(오픈 월드에서 사용 불가능) |
|인지적 구조 <br> (cognitive structure)| 단기/장기 메모리가 있음 <br> perceive-plan-act cycle 구조로 상황을 인지하고 <br> 그에 맞는 행동 수행 가능| 선행 지식(procedural knowledge)에 따라 만들어진 행동만 수행 가능 <br> 새로운 행동을 배우지 않음 (ex: FPS 게임)|
|LLM | LLM의 방대한 학습 데이터로 인해 사람 같은 답변을 수행할 수 있음 | 현재 상황에 대해서만 답변하므로 사람 같은 agent가 되기에는 부족한 부분이 있음 |

LLM만 단독 사용에서의 단점을 해결하기 위해서는 대용량의 과거 기억이 추가로 필요하나, 현재 LLM에는 context window 크기에 제한이 있다.

이를 해결하기 위해 이전 방법들에서는 언어 모델을 정적인 정보(static knowledge), 정보 검색(information retrieval), 요약 등의 방법으로 문제를 해결하려고 했다. 

$\Rightarrow$ 해당 논문은 매 time step마다 과거 경험이 현재 맥락과 계획까지 고려하여 동적으로 업데이트되는 방법을 제안한다. 

## 마을과 마을 주민(Agent) 만들기
- <b>마을</b>
    - tree 구조로 구현됨
    - parent node: "부엌" -- child node: "에어프라이어" $\Rightarrow$ "에어프라이어가 부엌에 있다"로 해석됨
    - 마을 주민이 해당 에어프라이어를 사용하게 된다면, 에어프라이어의 상태가 "idle" $\Rightarrow$ "치킨 데우기"로 바뀜
    - User가 직접 자연어로 상태를 바꿀 수 있음 (ex: "가스레인지에서 화재 발생"으로 상태를 바꾸면 근처에 있는 agent가 불났다고 인지 가능함)

- <b>Agent</b>
    - 초기 기억: 직업, 다른 agent와의 관계
    - Memory stream: 매 time step 행동이 저장됨. "생성된 시간 / 설명 / 최근 해당 기억을 접근한 시간"으로 구성됨
    - Agent environment tree: 마을 전체의 subgraph로, agent가 다른 곳을 탐색하면서 업데이트됨. Agent가 해당 장소에 없는 사이 바뀐 부분이 있다면, agent가 다시 방문할 때까지 해당 환경 트리는 업데이트가 안 됨

- <b>Agent 사이의 소통</b>
    - 자연어로 소통하며, 화면에서는 이모지로 간략하게 표현됨
    - agent들은 각자의 위치와 행동을 알고 있기에, 대화할지 말지를 LLM을 통해서 결정함

- <b>User와의 소통</b>
    - User가 agent와 소통하기 위해서는 역할(persona, 페르소나)이 필요함 (ex: 뉴스 리포터)
    - agent에게 명령을 전달할 때는 "해당 agent 내면의 목소리"로 역할을 설정하면 됨

- <b>Agent의 행동 위치</b>
    - 각 마을 주민의 행동 위치를 정할 때, 각 주민의 environment tree가 LLM에 전달됨
    - LLM은 전달된 값을 참고하여, 행동하기에 가장 적합한 tree의 leaf 노드를 찾음 

## Generative Agent 구조와 작동 원리

<figure class="centered-figure">
<img src="../assets/images/Generative-Agents/img_0002.jpeg"  style="width: 80%;">
<figcaption class="caption">Generative Agent 구조</figcaption>
</figure> 

- <b>Generative Agent의 목표</b>: 다른 agent와 상호작용하고 주변 환경의 변화에 반응하여 오픈 월드에서 작동하는 framework
- <b>논문의 목표</b>: 관련성이 가장 높은 agent의 기억을 가져오고 합성하는 구조를 만드는 것

### 1) Memory and Retrieval
<figure class="centered-figure">
<img src="../assets/images/Generative-Agents/img_0003.jpeg" style="width: 80%;">
<figcaption class="caption">Memory stream과 Retrieval 과정</figcaption>
</figure> 

- <b>목표</b>: 추후 판단에 사용할 적절한 경험 선정
- <b>"Retrieval Function"</b>: memory stream에서 기억을 선정하기 위한 판단 함수
    - <b>입출력</b>: 현재 상황 $\rightarrow$ memory stream의 부분 집합
    - <b>함수 주요 구성 요소 3가지</b>
        1. <b>Recency</b>: 최근 접근된 기억일수록 높은 점수, exponential decay 함수 사용 (factor 0.995)
        2. <b>Importance</b>: agent가 중요하다고 생각하는 기억일수록 높은 점수, 언어 모델에게 해당 기억의 중요한 정도를 물어봄
        3. <b>Relevance</b>: 현재 상황과 관련이 높은 기억일수록 높은 점수, 각 기억을 언어 모델로 embedding한 후, query 기억과의 cosine 유사도 점수 사용
    -  <b>최종 점수와 기억</b>: 각 요소 점수를 min-max 정규화한 후, weighted sum을 진행함. 논문에서는 weight를 다 1로 설정. 가장 점수가 높은 기억 N개가 선정되어, 언어 모델의 context window 안에 들어갈 수 있도록 함.

### 2) Reflection
<figure class="centered-figure">
  <img src="../assets/images/Generative-Agents/img_0004.jpeg" style="width: 80%;">
  <figcaption class="caption">Reflection의 결과물: Reflection Tree</figcaption>
</figure> 

- <b>목표</b>: 고차원적이며 더 추상적인 생각으로 발전시키기
- <b>방법</b>: 최근 이벤트에 대해서 importance 점수의 합이 일정 threshold를 넘으면 reflection을 주기적으로 진행
    1. 무엇을 성찰할지 결정하기:<br>
    LLM input: 최근 100개의 기억 + "해당 기억을 참고하여, 해당 주체에 대해 답변할 수 있는 가장 핵심적인 질문은 3가지는 무엇인가?" $\Rightarrow$ LLM output: 3가지 질문 생성<br>
    (ex: "최근 A가 빠져 있는 것은?", "A랑 B의 관계는 어떠한지?")
    2. 관련 기억 모으기:<br>
    위에서 생성한 질문을 query로 사용하여 관련된 기억(다른 reflection 포함)을 모음
    3. 인사이트 생성: <br>
    언어 모델에게 위의 경험을 토대로 인사이트와 각 인사이트의 근거가 되는 기억을 알려달라고 함
    4. memory stream에 저장: <br>
    인사이트(근거 기억에 대한 포인터 포함)를 저장

    <figure class="centered-figure">
      <img src="../assets/images/Generative-Agents/img_0005.jpeg" style="width: 75%;">
      <figcaption class="caption">Reflection 과정</figcaption>
    </figure>    

- <b>최종 결과물</b>: reflection tree 생성<br>
    - leaf: 기본적인 observation 
    - non-leaf: 위로 올라갈수록 더 추상적이고 고차원적인 생각

### 3) Planning and Reacting
- <b>목표</b>: 사람처럼 일관성 있는 행동하는 agent를 만들기 위한 장기적인 계획 수립. 순간적인 타이밍에 대해 optimize를 진행하면, 12시에 점심을 먹고 13시에 또 점심을 먹는 상황이 발생하기 때문에 장기적인 계획을 세우는 것이 중요함
- <b>구성 요소</b>: 장소, 시작 시각, 지속 시간
- <b>(a) 계획 세우는 방법</b>: 계획은 top-down 방식으로, 재귀적으로 더 구체적인 계획을 세움.
    1. 초기에 하루 계획을 폭넓게 설정:<br>
    LLM input: agent 설명 요약(이름, 특성, 최근 경험 요약) + 전날 계획 요약 + "오늘은 6월 14일로, A가 오늘 대략 할 일은?:    " $\Rightarrow$ LLM output: 해당하는 날의 대략적인 계획
    2. 대략적인 계획들을 memory stream에 저장
    3. 저장된 계획들을 재귀적으로 더 작은 행동으로 분해:<br>
    (ex: 13:00~17:00 작곡하기 <br>
    $\rightarrow$ 13:00 아이디어 구상하기 ... 16:00 마무리 작업하기 <br>
    $\rightarrow$ 13:00 작업실에서 컴퓨터 켜기 ... 16:50 작업실 정리하기)

- <b>(b) 반응하고 계획을 업데이트하는 방법</b>: agent들이 행동을 진행하는 동안에 환경이 변하게 되는데, 변하는 환경을 인식하고 계획을 계속할지 / 이에 대해 반응할지 결정하게 됨
    1. Agent의 상황 요약본 (Agent's Summary Description) 생성: <br>
        - 구성 요소: agent의 정보 + 주된 동기 부여 요인 + 현재 활동 + 자기 평가
        - 생성 과정: "agent 주요 특성은?" / agent의 현재 일상적인 활동"/ "agent의 최근 삶에 대한 느낌"을 query로 각각 retrieval 진행하여 관련 기억 수집 
        <br>$\rightarrow$ 수집된 기억을 언어 모델로 요약 
        <br>$\rightarrow$ agent의 정보와 각 query에 대한 대답 요약을 합쳐서 상황 요약본 생성
        - 자주 사용되므로 cache처럼 저장하여 사용
    2. 맥락 요약본 생성: "agent와 관찰된 대상과의 관계는?" / "관찰된 대상의 상태"에 대한 답변을 합쳐서 요약
    3. 상황과 맥락 요약본을 사용하여 agent가 취할 다음 행동을 언어 모델로 생성
    4. 반응을 하기로 했으면, 반응 시작 시각부터 이후 agent의 기존 계획 재생성
    5. 만약에 다음 행동이 "두 agent 간 상호작용 하기"일 시, 대화를 생성하게 됨

- <b>(c) 대화하기</b>
    - "agent 상태" + "관찰된 것" + <b>"각 agent가 서로를 기억하는 요약본"</b> + <b>대화 시작: "대화의 목적" / 대화 진행 중: "현재 대화 기록"</b>
    - 위의 구성 요소로 LLM이 다음 대사를 생성하고, 한 agent가 대화를 그만두기로 할 때까지 대화를 반복함

- <b>추가적인 특징</b> 
    - memory stream에 저장됨 
    - reflection 시에 plan 사용함
    - 필요시에 중간에 plan이 바뀔 수 있음

## 평가 및 결과
### 1) Controlled Evalutaion
-  <b>평가 부분</b>: 각 agent들이 사람같이 행동할 수 있는지를 평가하는 항목

- <b>평가 방법</b> 
    - Smallville에서 2일이 지난 후 agent와의 "인터뷰" 진행
    - 100명의 평가자가 아래 5개 조건에 대한 답변을 보고 "사람 같은지" 평가 진행

- <b>비교 조건</b>
    - 비교 시, <b>재시뮬레이션을 진행하지 않고 인터뷰 시점에서만 조건을 수정</b>하여 비교 진행
    - (a) <em>full architecture</em>: 인터뷰 진행 시, memory stream의 모든 기억 접근 가능
    - 인터뷰 진행 시, memory stream의 일부 기억 접근 제한
        - (b) <em>no observation, no reflection, no planning</em>
        - (c) <em>no reflection, no planning</em>
        - (d) <em>no reflection</em>
    - (e) <em>human crowdworker-authored behavior</em>: 각 agent의 행동을 지켜본 사람이 agent로서 질문 항목에 대한 답변 작성. 제안된 agent 구조가 기본적인 수준의 행동 역량을 갖추었는지 확인하기 위해 진행됨
    
- <b>결과</b>

    <figure class="centered-figure">
      <img src="../assets/images/Generative-Agents/img_0006.jpeg" style="width: 50%;">
      <figcaption class="caption">제안된 Generative Agent 구조에 관한 결과 비교</figcaption>
    </figure> 

    - memory stream의 전체 구조를 사용하는 조건 (a)가 가장 좋은 결과를 보였으며, 유의미한 차이를 보임
    - 인터뷰 중에 발생한 잘못된 답변의 종류
        1. observation 기억을 제한했을 때: 알고 지낸 agent 지인을 모른다고 함 $\rightarrow$ 제한 해제 후, 지인임을 알고 설명도 가능함
        2. 기억에서의 오류 발생: 다른 agent에게 들었던 내용을 기억하지 못하거나, 기억의 일부가 잘못된 경우 발생 (ex: 마을 이장 선거에 대해서 들었는데, 들은 적이 없다고 함 / 파티에서 선거에 관해 이야기하자는 것을 기억하나 파티 개최 여부를 확신하지 못함)
        3. 디테일 환각 (hallucinated embellishment) 발생<br>
            - 경험하지 않은 일을 꾸며내지 않음: 기억하지 못한다고 답변하지, 안 한 일을 했다고 말하지 않음 
            - 실제 경험한 일을 설명할 때 환각 발생: 경험에 관해 이야기하다가 실제 일어나지 않은 일들을 덧붙여 이야기함 (ex: 선거에 관해 이야기한 사실을 기억하나 "그가 내일 연설할 거야"와 같이 실제 계획하지 않은 일을 있었던 것처럼 설명함)
        4. 언어 모델에 내재한 지식이 잘못 나타남: 실제 존재했던 동명이인이 agent와 동일인이라고 생각하고 답변

### 2) End-to-End Evalutaion
-  <b>평가 부분</b>: agent의 안정성과 사회적 행동이 발생했는지에 대해 평가하는 항목
    - (a) Information Diffusion: 정보가 agent 사이에서 확산하는가? <br>(ex: A가 마을 이장 선거에 출마를 고려하는 중이라는 이야기가 퍼지는지?)
    - (b) Relationship Memory: agent가 새로운 다른 agent들과 관계를 형성하는가? <br>(ex: 공원에서 처음 대화한 상대와 친구가 되고, 나중에도 기억하고 반응하는지?)
    - (c) Coordination: 자율적인 사회적 상호작용이 가능한가? <br>(ex: 발렌타인데이 파티하고 싶다는 정보만으로 파티가 실제로 열릴지?)

- <b>평가 방법과 결과</b> 
    - <b>(a) Information Diffusion</b>
        - "Sam이 마을 이장 후보로 나왔다"와 "Isabella가 Hobbs 카페에서 발렌타인데이 파티를 연다"라는 정보가 널리 퍼졌는지 확인
            - 시뮬레이션 초기 설정에 "Sam"과 "Isabella"에게 각각의 정보를 준 후, 이틀이 지난 후에 25명의 마을 주민 agent와 인터뷰 진행
            - agent가 위 두 사실을 알고 있는지 (Yes/No), 알고 있다면 알게 된 대화 내역까지 확인 (hallucination)
            - 시뮬레이션이 끝난 후, 정보를 가지고 있는 agent의 퍼센트로 평가
        - 결과
            - "Sam이 마을 이장 후보로 나왔다": 4% (1명) $\Rightarrow$ 32% (8명)
            - "Isabella가 Hobbs 카페에서 발렌타인데이 파티를 연다": 4% (1명) $\Rightarrow$ 52% (13명)
            - 해당 사실을 모르는데 안다고 말한 agent는 없었음

        - <b>(b) Relationship Memory</b>
            - 인터뷰를 통해 마을 주민 간의 undirected graph 관계도 생성
                - vertices(V): agent / edge(E): 서로 아는 사이일 경우 연결됨
                - $\vert V \vert$와 $\vert E \vert$는 각각의 개수
                - network density 계산: $\eta = \frac{총\ 연결\ 수}{가능한\ 연결\ 수} = 2 * \frac{\vert E\vert}{\vert V\vert(\vert V\vert -1)}$
            - 결과
                - 밀도: 0.167 $\Rightarrow$ 0.74
                - 453개의 응답 중 6개 (1.3%)에서 hallucination 발생

        - <b>(c) Coordination</b>
            <figure class="centered-figure">
            <img src="../assets/images/Generative-Agents/img_0007.png" style="width: 80%;">
            <figcaption class="caption">발렌타인데이 파티를 알고 있는 agent들</figcaption>
            </figure> 

            - "Isabella"가 주최한 발렌타인데이 파티를 듣고 실제로 나타난 agent 개수 확인
            - 결과
                - 파티에 대해 들은 12명 중 5명이 파티에 참석함
                - 불참석자 인터뷰: 바쁜 일이 있는 등 파티 참석을 갈등함(3명), 관심은 있었으나 파티 당일에 참석하는 것을 

### 3) 실험의 한계와 에러
- 기억(메모리)이 많아질수록 retrieving이나 행동을 할 장소 선정에 어려움 발생
- 적합한 행동을 잘 판단하지 못함<br>
(ex: 기숙사 방 안의 화장실에 사용하는 agent가 있는데 다른 agent가 동일한 화장실에 들어가는 상황 발생 <br>
$\rightarrow$ 이는 통상적으로 기숙사 화장실은 다수가 사용하도록 설계되어 있기에 이를 바탕으로 행동한 결과 <br>
$\rightarrow$ "dorm bathroom"을 "one-person bathroom"으로 변경하여 해결)
- instruction tuning에 대한 잠재적인 가능성 확인<br>
(ex: Mei가 공손한 말투로 대화를 시작하기에, 남편과의 대화가 형식적으로 흘러감 / Isabella는 너무 협조적이여서, 다른 agent에게 다양한 아이디어를 받으면서 본인의 흥미가 달라짐)

## 저자들의 향후 계획과 느낀 한계
1. 함수 고도화: retrieval 세부 함수들을 fine-tuning 해서 더 적절한 경험 선정하는 방식 등으로 함수 발전시키기
2. 비용 절약: real-time 상호작용까지 발전할 수 있도록 언어 모델 개발
3. 평가 지표: 더 긴 시간의 시뮬레이션이 진행된 상황에서의 평가와 성능 측정을 위한 벤치마크가 필요함
4. 모델의 견고함: prompt나 메모리 해킹, hallucination에 얼마나 취약할지 파악이 필요함
5. 언어 모델의 불완전성: 언어 모델의 편향성이나 생소한 일에 대해 잘 생성하지 못하는 문제 존재 
<br><br>

# 🔍 개인적인 생각
<!-- 줄글로 작성 (내 해석/비판적 시각 강조)
예시:
이 논문은 제안한 방법으로 [문제 A]를 잘 해결하지만, [상황 X]에서는 성능 저하 가능성이 있습니다. 
또한 논문에서 다루지 않은 [측면 Y]에 대한 고려가 필요하다고 생각합니다. 
실제로 제가 관심 있는 [적용 분야]에서는 이러한 점을 주의해야 할 것 같습니다.
-->
논문을 읽고 2가지 생각이 들었다.<br>

먼저, 해당 방법에서 각 agent의 행동과 생각 과정을 전부 자연어로 기록했기 때문에 여러 문제가 따랐을 것 같다는 점이다. 만약 API가 아닌 자체 언어 모델을 활용해 자연어 대신 embedding 값을 사용했다면 더 효율적으로 문제를 해결할 수 있지 않았을까라는 생각이 들었다. 물론 embedding 값을 다시 자연어로 변환해 검토하는 과정이 필요하고, 해당 embedding 번역 모델의 정확도도 검증해야 하니 초창기 논문으로는 당연한 부분인 것 같다.<br>
이 논문은 2023년에 발표되었고, 이후에 많은 문제를 해결한 논문들이 많이 나왔을 것이다. 실제로 연구를 넘어 이를 활용한 게임도 나왔으니 말이다…! (ex: 크래프톤의 "인조이", 크래프톤 산하 스튜디오 ReLU Games의 "언 커버 더 스모킹건(hallucination까지도 게임 요소로 활용한 아이디어가 너무 존경스럽다….)")<br>

또한, 일레븐랩스에서 두 개의 음성 AI assistant가 서로 언어 모델임을 인식한 순간부터 자연어가 아닌 "사림이 이해 불가능한 소리"로 효율적으로 소통하도록 만든 [[GibberLink](https://elevenlabs.io/blog/what-happens-when-two-ai-voice-assistants-have-a-conversation)]가 떠올랐다. 결국 자연어가 아닌 기계 간에 최적화된 새로운 언어가 생겨나는 것은 자연스러운 흐름이 아닐까 싶다.<br>

다음으로 인상 깊었던 점은, 논문에서 "글 쓰는 스타일"에 의해 instruction tuning이 일어났고 결과적으로 너무 형식적으로 대화하는 agent가 나왔다는 설명이다. 이를 보면서, LLM이 점점 스마트폰처럼 일상에 없어서는 안 될 기술로 자리 잡고 있는 시대에 가장 각광받는 능력은 뛰어난 <b>"글쓰기"</b>라는 생각이 들었다. ChatGPT를 비롯한 다양한 LLM 모델을 사용하여 우리가 원하는 완벽한 대답을 얻어내기 위해선, "잘 정리된 사전 정보를 제공하고, 질문을 기가 막히게 잘해야" 한다는 것은 모두가 경험했을 것이다. 모델이 아무리 발전해도 "사전 정보와 질문"을 잘해야 한다는 사실엔 변함이 없을 것 같다. 결국 이를 구헌하는 것은 글쓰기와 이를 뒷받침하는 능력일 것이다!

<!-- # 💡 내 연구/프로젝트에 어떻게 활용할까? -->
<!-- 줄글로 작성 (내 연구/프로젝트에의 적용 관점)
예시:
현재 진행 중인 [프로젝트명]에서는 이 논문의 [기법/아이디어]를 [부분/전체] 적용할 수 있을 것으로 보입니다. 
특히 [구체적 상황]에서는 큰 도움이 될 것 같고, 반면 [다른 상황]에서는 기존 방법과의 trade-off를 고려해야 할 것 같습니다.
-->

<!-- # 📚 참고할 만한 연구 및 추가로 읽을 것 -->
<!-- Bullet 형태로 작성 (연결 논문, 추후 읽을 논문)
예시:
- [논문명1] - [간단한 관련성 설명]
- [논문명2] - [간단한 관련성 설명]
- Follow-up: [키워드/논문명3] 추가로 읽어볼 계획
-->

<!-- # ✏️ 개인 메모 / 앞으로 할 일 -->
<!-- Bullet 형태로 작성 (개인 노트용, 자유롭게)
예시:
- [부분 A]는 아직 이해가 부족함 → 추가 복습 필요
- 실험 시 [파라미터 B]가 성능에 미치는 영향 확인해볼 것
- [팀원/멘토]와 이 논문에 대해 토론해보기
-->
