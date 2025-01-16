import keras
from lstm02 import model2, lstr_padd, lstr_target, lsval_padd, lsval_target, model1, model3, lstest_input, lstest_target
from set_data import set_padding
from set_model import myfit_lstm
from util import draw_plot, df_maker

keras.utils.set_random_seed(5)
# fit-> 그래프 생성
# legends = ['train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']
# #his1 = myfit_lstm(model1, "lstm01", lstr_padd, lstr_target, lsval_padd, lsval_target)
# #his2 = myfit_lstm(model2, "lstm02", lstr_padd, lstr_target, lsval_padd, lsval_target)
# his3 = myfit_lstm(model3, "lstm03", lstr_padd, lstr_target, lsval_padd, lsval_target)
# data_list=[his3.history['accuracy'], his3.history['loss'], his3.history['val_accuracy'], his3.history['val_loss']]
# draw_plot('lstm03.png', data_list, 'epoch', 'accuracy', 'LSTM03-dropout0.3', legends)

# test 검증 및 표 생성
test_padd = set_padding(100, lstest_input, 'pre')
result1 = keras.models.load_model('best_lstm01_model.keras')
result2 = keras.models.load_model('best_lstm02_model.keras')
result3 = keras.models.load_model('best_lstm03_model.keras')

col_list = ['accuracy', 'loss']
index_list=['LSTM_basic', 'LSTM_dropout', 'LSTM_layer_add']
con = []
test1_his = result1.evaluate(test_padd, lstest_target)
con.append([test1_his[0], test1_his[1]])
test2_his = result2.evaluate(test_padd, lstest_target)
con.append([test2_his[0], test2_his[1]])
test3_his = result3.evaluate(test_padd, lstest_target)
con.append([test3_his[0], test3_his[1]])

df = df_maker(col_list, index_list, con)
df.to_csv('lstm_test.csv')
print(f"LSTM_basic: {test1_his}")
print(f"LSTM_dropout: {test2_his}")
print(f"LSTM_layer_add: {test3_his}")
