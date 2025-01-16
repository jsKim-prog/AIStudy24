# 데이터셋 : tr-val(80:20), 시퀀스 패딩 100
import keras
from tensorflow.python.keras.utils.version_utils import callbacks

from set_data import load_imdb, div_data, set_padding

lsdata_ip, lsdata_target, lstest_input, lstest_target = load_imdb(300)
lstr_input, lsval_input, lstr_target, lsval_target = div_data(lsdata_ip, lsdata_target)
print(f"훈련/검증/테스트입력 : {lstr_input.shape}/{lsval_input.shape}/{lstest_input.shape}")
print(f"훈련/검증/테스트타겟 : {lstr_target.shape}/{lsval_target.shape}/{lstest_target.shape}")

lstr_padd = set_padding(100, lstr_input, 'pre')
lsval_padd = set_padding(100, lsval_input, 'pre')
# for i in lstr_padd[:10]:
#     print(i[:10])

# 모델생성용 공통층 정의
embedd_layer = keras.layers.Embedding(300, 16, input_shape=(100,))
lstm_layer1 = keras.layers.LSTM(8)
lstm_layer2 = keras.layers.LSTM(8, dropout=0.3)
lstm_layer_re =keras.layers.LSTM(8, dropout=0.3, return_sequences=True)
outputs = keras.layers.Dense(1, activation='sigmoid')
# 컴파일, fit 공통 함수 정의
def myfit(model, name, input_data, target_data, val_input_data, val_target_data):
    save_name = "best_"+name+"_model.keras"
    rsmprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=rsmprop,loss='binary_crossentropy', metrics=['accuracy'])
    save_point=keras.callbacks.ModelCheckpoint(save_name, save_best_only=True)
    stop_point=keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(input_data, target_data, epochs=100, batch_size=64, validation_data=(val_input_data, val_target_data), callbacks=[save_point, stop_point])
    return history

model_lstm1 = keras.Sequential()
model_lstm1.add(embedd_layer)
model_lstm1.add(lstm_layer1)
model_lstm1.add(outputs)
model_lstm1.summary()


model_lstm2 = keras.Sequential()
model_lstm2.add(embedd_layer)
model_lstm2.add(lstm_layer2)    # dropout 0.3
model_lstm2.add(outputs)
model_lstm2.summary()

model_lstm3 = keras.Sequential()
model_lstm3.add(embedd_layer)
model_lstm3.add(lstm_layer_re)  #return_sequences=True /dropout 0.3
model_lstm3.add(lstm_layer2)
model_lstm3.add(outputs)
model_lstm3.summary()


keras.utils.plot_model(model_lstm3, show_shapes=True, show_layer_activations=True, to_file='model_lstm03.png')

# 결론 : keras.layer 는 새로운 모델에는 반드시 새로운 객체생성하여 붙여야 함
# 변수로 저장하여 기존 객체 불러오는 경우 param 변동이 안됨