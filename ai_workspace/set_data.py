import keras
from sklearn.model_selection import train_test_split

# KERAS fashion_mnist dataset 받기
def load_mnist():
    from keras.src.datasets import fashion_mnist
    (data_input, data_target), (test_input, test_target) = fashion_mnist.load_data()
    print(f"훈련데이터/타겟 : {data_input.shape} / {data_target.shape}")
    print(f"테스트데이터/타겟 : {test_input.shape} / {test_target.shape}")
    return data_input, data_target, test_input, test_target

# 데이터셋 가져오기-> imdb.com 리뷰-> data_input, test_input
def load_imdb(dic_size):
    from keras.src.datasets import imdb
    (data_ip, data_tg), (test_ip, test_tg) = imdb.load_data(num_words=dic_size)
    print(f"훈련데이터/타겟 : {data_ip.shape} / {data_tg.shape}")
    print(f"테스트데이터/타겟 : {test_ip.shape} / {test_tg.shape}")
    return data_ip, data_tg, test_ip, test_tg

def make_2d_data(train_input):
    x_num = train_input.shape[1]    # 해상도 가로
    y_num = train_input.shape[2]    # 해상도 세로
    tr_2d_data = train_input.reshape(-1, x_num*y_num)
    return tr_2d_data

def make_scaled(train_input):
    scaled = train_input/255.0
    return scaled

def div_data(train_input, train_target):
    tr_input, val_input, tr_target, val_target = train_test_split(train_input, train_target, test_size=0.2)
    print(f"훈련데이터/타겟 : {tr_input.shape} / {tr_target.shape}")
    print(f"검증데이터/타겟 : {val_input.shape} / {val_target.shape}")
    return tr_input, val_input, tr_target, val_target

# 시퀀스 패딩
def set_padding(max_len, data_arr, trc_str):
    from keras.api.preprocessing.sequence import pad_sequences
    padded_data = pad_sequences(data_arr, maxlen=max_len, truncating=trc_str)
    return padded_data

