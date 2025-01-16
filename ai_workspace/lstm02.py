import keras
from set_data import load_imdb, div_data, set_padding
from set_model import make_model_lstm

dic_size=500
lsdata_ip, lsdata_target, lstest_input, lstest_target = load_imdb(dic_size)
lstr_input, lsval_input, lstr_target, lsval_target = div_data(lsdata_ip, lsdata_target)
print(f"훈련/검증/테스트입력 : {lstr_input.shape}/{lsval_input.shape}/{lstest_input.shape}")
print(f"훈련/검증/테스트타겟 : {lstr_target.shape}/{lsval_target.shape}/{lstest_target.shape}")

lstr_padd = set_padding(100, lstr_input, 'pre')
lsval_padd = set_padding(100, lsval_input, 'pre')

model1 = make_model_lstm(dic_size, [keras.layers.LSTM(8)])
model2=make_model_lstm(dic_size, [keras.layers.LSTM(8, dropout=0.3)])
model3=make_model_lstm(dic_size, [keras.layers.LSTM(8, dropout=0.3, return_sequences=True),keras.layers.LSTM(8, dropout=0.3)])

model1.summary()
model2.summary()
model3.summary()