import tensorflow as tf
import keras
from set_data import *

# 재현성 관련 설정
keras.utils.set_random_seed(5)
tf.config.experimental.enable_op_determinism()
# 데이터셋 제작
(data_input, data_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = make_scaled(data_input)  # 0~1사이값으로 표준화
train_scaled, mul_resol = make_2d_data(train_scaled)    # 해상도 변환
tr_input, val_input, tr_target, val_target = div_data(train_scaled, data_target)
 # 검증데이터 20%
print(f"훈련데이터/타겟 : {tr_input.shape} / {tr_target.shape}")   # 훈련데이터/타겟 : (48000, 28, 28) / (48000,)
print(f"테스트데이터/타겟 : {test_input.shape} / {test_target.shape}")  # 테스트데이터/타겟 : (10000, 28, 28) / (10000,)
print(f"검증데이터/타겟 : {val_input.shape} / {val_target.shape}") # 검증데이터/타겟 : (12000, 28, 28) / (12000,)


# 모델생성
  # 입력-> 출력(소프트맥스: 다중분류)
dense = keras.layers.Dense(10, activation='softmax', input_shape=(mul_resol,))  # 출력층
model = keras.Sequential([dense])
model.summary()

# compile -> fit -> 검증
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(tr_input, tr_target, epochs=5)
val_score = model.evaluate(val_input, val_target)
# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss, accuracy')
plt.show()
print(f"검증점수 : {val_score[1]:.2%}")
# 2차 : 검증점수 : 84.38%
