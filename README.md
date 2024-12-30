![header](https://capsule-render.vercel.app/api?type=waving&height=150&color=gradient&text=AI%20Study&desc=파이썬을%20이용한%20머신러닝/딥러닝%20학습용&fontColor=1B1833&descSize=-3&fontAlignY=28)

## 시나리오
MBC 상점에서 앱마켓을 운영하는데 AI를 활용하는 기법을 학습해 보자.
* 1단계 : 앱마켓에서 살아있는 생선을 팔기 시작했다.
* 2단계 : 물류센터에서 생선을 고르는 직원이 있는데 생선이름을 외우지 못한다.
* 생선의 종류 : 도미, 곤들매기, 농어, 강꼬치고기, 빙어, 송어.....
* AI 미션 : 생선의 길이가 30cm 이상이면 도미!!!

## 머신러닝 기초
*  File :  [머신러닝_생선길이_학습](https://github.com/jsKim-prog/AIStudy24/blob/master/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EC%83%9D%EC%84%A0%EA%B8%B8%EC%9D%B4_%ED%95%99%EC%8A%B5.ipynb) / [훈련세트_테스트세트](https://github.com/jsKim-prog/AIStudy24/blob/master/%ED%9B%88%EB%A0%A8%EC%84%B8%ED%8A%B8_%ED%85%8C%EC%8A%A4%ED%8A%B8%EC%84%B8%ED%8A%B8.ipynb)
*  AI 훈련 기본 단계
   1. 📈 데이터 전처리 및 훈련데이터 생성
   2. 🧩 AI 트레이닝
   3. 📋 훈련 평가

### 📈 데이터 전처리 및 훈련데이터 생성
> 훈련 데이터(training data)
> - 입력(input) : 훈련용 데이터
> - 타겟(target) : 정답 데이터
> 
> 특성(feature) : 입력(input)에 사용된 속성(예 : 길이, 무게)
  
* `import matplotlib.pyplot as plt`
  - `matplotlib` : 과학계산용 그래프 그리기용 패키지 (관례 : as mlt)
  - `pyplot` : matplotlib의 하위 API를 포장(wrapping)한 명령어 집합을 제공(관례 : as plt)

* `zip()` : 여러 개의 순회가능한 객체를 받아 각 객체의 원소를 차례로 묶어 튜플로 반환
  ```python
  a = [1, 2, 3]
  b = [‘a’, ‘b’, ‘c’]
  for pair in zip(a, b):
     print(pair)

  (1, ‘a’)
  (2, ‘b’)
  (3, ‘c’)
  ```

* `import numpy as np`
     - Numpy : 파이썬의 대표적인 배열 라이브러리로, 고차원 배열을 손쉽게 조작 -> 샘플링 편향을 막기 위해 사용
     - 관례 : as np
     - `.shape` : 배열의 크기를 알려주는 메서드
       
       ```python
        print(input_arr.shape) 
        (49, 2)   # 2개의 열(길이, 무게), 49행(데이터 49개)
        ```

### 🧩 AI 트레이닝
* `from sklearn.neighbors import KNeighborsClassifier`
   - `sklearn` : 사이킷런 패키지를 활용한 AI 트레이닝
   - `KNeighborsClassifier` : k-최근접 이웃 알고리즘 사용
   - > k-최근접 이웃 알고리즘
     > - 어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고 다수를 차지하는 것을 정답으로 출력 
     > - 객체를 메모리에 만들고 활용 -> 데이터가 커지면 메모리가 많이 필요하며, 직선거리 계산시간도 오래 걸림
   - `kn = KNeighborsClassifier() `
     - 객체 생성하여 변수 연결(java : KNeighborsClassifier kn = new KNeighborsClassifier())
     - 참조데이터 설정 가능(default : 5개)

### 📋 훈련 평가
* `.score()`
   - 객체의 훈련정도 평가
   - 1.0 == 100%(정확도)

* `. predict([[new_length, new_weight]])`
  - 새로운 데이터의 정답을 예측
  - 데이터가 2차원 배열 형태로 입력됨 -> 새로운 데이터도 2차원 배열 형태로 입력     
