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
