# 합성곱 사용 실습
# data : keras의 fashion_mnist
# 표준화, non-flatten data 사용
# 함수API 이용

import keras
from sklearn.model_selection import train_test_split
from keras.src.utils import plot_model

keras.utils.set_random_seed(32)

#데이터 준비
(data_input, data_target),(test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = data_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, data_target, test_size=0.2) # 검증용 데이터 20% 분리
print(f"훈련데이터 : {train_scaled.shape}")

# 모델 생성(함수형 api 사용)
# 합성곱1(Conv2D->maxPooling : layer[0])
# 합성곱2(Conv2D->maxPooling : layer[1])
# -> Flatten -> dense(hidden : relu)-> Dropout(0.3) -> dense(output:softmax)

cnn_input = keras.Input(shape = (28, 28, 1))
conv_layer1 = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')
pooling = keras.layers.MaxPooling2D(2)
#pooling2 = keras.layers.MaxPooling2D(2) # 같은 레이어, 변수 돌려쓰기 안됨(한 레이어에서 중복실행됨)
conv_layer2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')
flatten = keras.layers.Flatten()
dense_hd = keras.layers.Dense(100, activation='relu')
dense_do = keras.layers.Dropout(0.3)
dense_out = keras.layers.Dense(10, activation='softmax')

layer_1 = conv_layer1(cnn_input)
layer_2 = pooling(layer_1)
layer_3 = conv_layer2(layer_2)
layer_4 = pooling(layer_3)
layer_5 = flatten(layer_4)
layer_6 = dense_hd(layer_5)
layer_7 = dense_do(layer_6)
cnn_output = dense_out(layer_7)

model_cnn = keras.Model(cnn_input, cnn_output)

# 모델 시각화(테스트 완료 후 주석처리)
# model_cnn.summary()
plot_model(model_cnn, to_file="model_cnn02.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
# plot_model : graphviz 설치 필수(https://graphviz.gitlab.io/download/)
# to_file="model_cnn.png" : 프로젝트 폴더에 이미지 저장

# 컴파일, 훈련
# model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# check_point = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',save_best_only=True)
# early_stop_point = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
# his_cnn = model_cnn.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[check_point, early_stop_point])