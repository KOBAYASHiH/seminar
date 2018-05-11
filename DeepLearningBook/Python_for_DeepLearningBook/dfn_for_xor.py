import os
import sys
import json
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input


class XOR:
    # model
    def nn(input_len=2, hidden_len=2, output_len=1):

        input = Input(shape=(input_len,))

        h1 = Dense(hidden_len,
                   activation='relu',
                   name='hidden_1')(input)

        output = Dense(output_len,
                       activation='linear',
                       name="output",
                       use_bias=False)(h1)

        model = Model(inputs=input, outputs=output, name="XOR")
        model.compile(optimizer='sgd', loss='mse',
                      metrics=['accuracy'])
        return model

    # train
    def train(X, y, out_dir, train_ratio, epochs, batch_size):

        tb_cb = keras.callbacks.TensorBoard(log_dir=out_dir, histogram_freq=1)
        weights_filepath = os.path.join(out_dir, model_name) + '.h5'
        cp_cb = keras.callbacks.ModelCheckpoint(weights_filepath,
                                                save_best_only=True)
        model = XOR.nn()
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  validation_split=1.0 - train_ratio,
                  callbacks=[tb_cb, cp_cb])

        model.save_weights(os.path.join(out_dir, model_name + '_last.h5'))

        score = model.evaluate(X, y)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    # test
    def test(X, y, out_dir, model_name):
        # model load
        model = XOR.nn()
        model.summary()
        model.load_weights(os.path.join(out_dir, model_name + '.h5'))
        print(model.get_weights())

        collect_num = 0
        data_num = len(y)

        # result = model.test_on_batch(X, Y)
        predictor = model.predict(X)
        print(predictor)
        print(y)

        for idx in range(data_num):
            if idx < data_num:
                if np.allclose(y[idx], predictor[idx]):
                    collect_num += 1
                else:
                    collect_num += 0
            else:
                break

        test_accuracy = collect_num / data_num
        print(test_accuracy)


if __name__ == "__main__":

    # argvs = sys.argv
    json_file = 'configs/XOR.json'

    if os.path.splitext(json_file)[1] != '.json':
        sys.exit("ERROR : please input json path")

    f = open(json_file, 'r')
    json_data = json.load(f)

    out_dir = json_data["out_dir"]
    model_name = json_data["model_name"]
    train_ratio = json_data["train_ratio"]
    epochs = json_data["epochs"]
    batch_size = json_data["batch_size"]

    X = np.array([[True, True],
                  [True, False],
                  [False, True],
                  [False, False]],
                 dtype=bool)

    y = np.array([False, True, True, False],
                 dtype=bool)

    XOR.train(X, y, out_dir, train_ratio, epochs, batch_size)
    XOR.test(X, y, out_dir, model_name)

    f.close()
