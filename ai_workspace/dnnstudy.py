import tensorflow as tf
import keras
import numpy as np
import set_data

# 재현성 관련 설정
keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

# 데이터셋 : 표준화, no-flatten
input_d = set_data.train_input
target_d = set_data.train_target
input_v = set_data.vali_input
target_v = set_data.vali_target
resol_x = set_data.x_res  # 해상도
resol_y = set_data.y_res
print(input_d.shape, input_v.shape)
print(resol_x, resol_y)

# 모델생성 : 입력+은닉1+출력

# sig100 : sigmoid함수, param 100
# sig50 : sigmoid함수, param 50
# relu100 : relu함수, param 100
# relu50 : relu함수, param 50
flatten = keras.layers.Flatten(input_shape=(resol_x, resol_y))    # 플래튼층
outlayer = keras.layers.Dense(10, activation='softmax') # 출력층
sig50 = keras.layers.Dense(50, activation='sigmoid')
scores={"sig100":[],
        "sig50":[],
        "relu100":[],
        "relu50":[]}
model_sig50 = keras.Sequential([flatten, sig50, outlayer], name='sig50')
model_sig50.summary()
model_sig50.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
his2=model_sig50.fit(input_d, target_d, epochs=5)
scores["sig50"] = his2

import matplotlib.pyplot as plt
plt.plot(his2.history['loss'])
plt.plot(his2.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss, accuracy')
plt.show()