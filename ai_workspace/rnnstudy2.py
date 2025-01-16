import keras
import numpy as np
from set_data import load_imdb, div_data
import matplotlib.pyplot as plt
keras.utils.set_random_seed(5)

dic_size = 500  # 어휘사전 크기
def set_padding(max_len, data_arr, trc_str):
    from keras.api.preprocessing.sequence import pad_sequences
    padded_data = pad_sequences(data_arr, maxlen=max_len, truncating=trc_str)
    return padded_data
# 데이터셋 가져오기-> imdb.com 리뷰-> data_input, test_input
tr_input, tr_target, ts_input, ts_target = load_imdb(dic_size)
train_input, val_input, train_target, val_target = div_data(tr_input, tr_target)    # 20% 검증데이터 분리
# 데이터셋 분석(단어길이 평균, 최대, 최소, 중간값, 빈도(그래프))
word_lengths = np.array([len(x) for x in tr_input])
print(f"리뷰단어 길이 평균 : {np.mean(word_lengths)}")
print(f"리뷰단어 최대길이 : {np.max(word_lengths)}")
print(f"리뷰단어 최소길이 : {np.min(word_lengths)}")
print(f"리뷰단어 길이 중간값 : {np.median(word_lengths)}")

print(train_input[0])
print(len(train_input[0]))

# plt.hist(word_lengths)
# plt.xlabel('length')
# plt.ylabel('frequency')
# plt.show()

# 데이터 시퀀스 통일(100)
tr_padding = set_padding(100, train_input, 'pre')
val_padding = set_padding(100, val_input, 'pre')
print(f"패딩적용 사이즈(훈련/검증) : {tr_padding.shape} / {val_padding.shape}")
# print(tr_padding[5])

# 순환신경망만들기

# rnn2 : 원핫인코딩 사용
tr_oh = keras.utils.to_categorical(tr_padding)
val_oh=keras.utils.to_categorical(val_padding)
model_rnn2 = keras.Sequential()
model_rnn2.add(keras.layers.SimpleRNN(8, input_shape=(100, 500), activation='tanh'))
model_rnn2.add(keras.layers.Dense(1, activation='sigmoid'))
# 훈련(최적화 : RMSprop)
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model_rnn2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
save_point_rnn2 = keras.callbacks.ModelCheckpoint('best-rnn2-model.keras', save_best_only=True)
early_stop_point = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
his_rnn2 = model_rnn2.fit(tr_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[save_point_rnn2, early_stop_point])

plt.plot(his_rnn2.history['loss'])
plt.plot(his_rnn2.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.show()