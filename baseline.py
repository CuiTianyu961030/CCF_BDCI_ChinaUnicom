import pandas as pd
import numpy as np
from numpy import argmax
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import csv
import random

def build_data():
    train_data = np.array(pd.read_csv("train_all.csv"))
    predict_data = np.array(pd.read_csv("republish_test.csv"))

    # data_reshuffle(train_data)
    # data_reshuffle(test_data)
    # train_data = np.delete(train_data, [165491, 273702, 316212, 349037, 456036, 459053], axis=0)
    # predict_data = np.delete(predict_data, [131428, 170278], axis=0)

    x_train = train_data[:, :25]
    y_train = train_data[:, 25]
    x_predict = predict_data[:, :25]
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=32)
    user_id = predict_data[:, 25]

    # return x_train, y_train, x_test, y_test, x_predict, user_id
    return x_train, y_train, x_predict, user_id

def data_reshuffle(raw_data):
    default_element_row = []

    for x in range(0, len(raw_data[:, 0])):
        for y in range(0, len(raw_data[0, :25])):
            if raw_data[x, y] == "\\N":
                default_element_row.append(x)
    raw_data = np.delete(raw_data, default_element_row, axis=0)

    return raw_data

# def train_and_test(x_train, y_train, x_test, y_test, x_predict):
def train_and_test(x_train, y_train, x_predict):

    # model = Sequential()
    # model.add(Conv1D(32, 1, activation='relu', input_dim=25))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # # model.add(Dropout(0.25))
    # model.add(Conv1D(64, 3, activation='relu'))
    # model.add(Conv1D(64, 3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(11, activation='softmax'))
    # model = Sequential()
    # model.add(Dense(12, activation='relu', input_shape=(25,)))
    #
    # model.add(Dense(8, activation='relu'))
    #
    # model.add(Dense(1, activation='sigmoid'))
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)
    #
    # score = model.evaluate(x_test, y_test, batch_size=16, verbose=1)


    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    # model.add(Dense(30, activation='relu', input_dim=25))
    # model.add(Dense(40, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(40, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=10,
              batch_size=128)
    # score = model.evaluate(x_test, y_test, batch_size=128)
    # print(score)

    y_predict = model.predict(x_predict)

    # model.save("DNN_2018_10_16.h5")

    return y_predict

def metric(y_predict, y_test):

    y_predict = y_predict.astype(int)  # 转化成整型
    print(confusion_matrix(y_test, y_predict))

    precision = precision_score(y_test, y_predict)
    print(precision)

    recall = recall_score(y_test, y_predict)
    print(recall)

    f1 = f1_score(y_test, y_predict)
    print(f1)

if __name__ == "__main__":

    # x_train, y_train, x_test, y_test, x_predict, user_id = build_data()
    x_train, y_train, x_predict, user_id = build_data()

    service_num_dict = {}
    y_train_new = []
    y_test_new = []
    transfer_number = 0
    # print(y_train[0])
    for service_num in y_train:
        if service_num not in service_num_dict.keys():
            service_num_dict[service_num] = transfer_number
            transfer_number += 1
        y_train_new.append(service_num_dict[service_num])

    # for service_num in y_test:
    #     if service_num not in service_num_dict.keys():
    #         service_num_dict[service_num] = transfer_number
    #         transfer_number += 1
    #     y_test_new.append(service_num_dict[service_num])

    classes = len(service_num_dict)
    # print(service_num_dict)
    y_train = keras.utils.to_categorical(y_train_new, num_classes=classes)
    # y_test = keras.utils.to_categorical(y_test_new, num_classes=classes)
    # print(y_train)
    # model = load_model("DNN_2018_10_16.h5")
    # y_predict = model.predict(x_predict)

    # y_predict = train_and_test(x_train, y_train, x_test, y_test, x_predict)
    y_predict = train_and_test(x_train, y_train, x_predict)
    # print(y_predict)
    predict = []
    for element in y_predict:
        # print(argmax(element))
        predict.append(list(service_num_dict.keys())[int(argmax(element))])
    # metric(y_predict, y_test)
    print(predict)

    f = open("submission.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["user_id", "current_service"])
    for i in range(0, len(user_id)):
        writer.writerow([user_id[i], predict[i]])
        # if i != 131428 or i != 170278:
        #     writer.writerow([user_id[i], y_predict[i]])
        # else:
        #     writer.writerow([user_id[i], random.choice(y_train)])
    f.close()






