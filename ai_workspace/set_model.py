import keras
# 모델생성 등 공통함수
def make_model_lstm(dic_size, layers):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(dic_size, 16, input_shape=(100,)))
    if len(layers)==0:
        pass
    else:
        for i in layers:
            model.add(i)
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

# 컴파일, fit 공통 함수 정의
def myfit_lstm(model, name, input_data, target_data, val_input_data, val_target_data):
    save_name = "best_"+name+"_model.keras"
    rsmprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=rsmprop,loss='binary_crossentropy', metrics=['accuracy'])
    save_point=keras.callbacks.ModelCheckpoint(save_name, save_best_only=True)
    stop_point=keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(input_data, target_data, epochs=100, batch_size=64, validation_data=(val_input_data, val_target_data), callbacks=[save_point, stop_point])
    return history