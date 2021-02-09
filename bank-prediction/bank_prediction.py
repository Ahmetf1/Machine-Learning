import numpy as np
import csv
from keras.models import Sequential
from keras import layers, optimizers, losses
import pickle

class Pipeline():
    def __init__(self,data=[]):
        self.data=data
    def preprocess(self,new_list):
        new_list = np.array(new_list)
        new_list = new_list[1:, 3:]
        x = new_list[:, :-1]

        gender_data = []
        for gender in new_list[:, 2]:
            if gender == "Female":
                gender_data.append([1])
            else:
                gender_data.append([0])
        x_new = np.append(x, gender_data, axis=1)

        country_data = []
        last_country_data=[]
        row = []
        for country in new_list[:, 1]:
            if not country in country_data:
                country_data.append(country)
        try:
            for country in new_list[:, 1]:
                if country_data.index(country) == 0:
                    row = [0, 0]
                if country_data.index(country) == 1:
                    row = [0, 1]
                if country_data.index(country) == 2:
                    row = [1, 1]
                last_country_data.append(row)
        except:
            print("there are more than 3 country please edit the code")

        x_new = np.append(x_new, last_country_data, axis=1)
        x = x_new[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        y_last = new_list[:, -1]
        x_last=[]

        self.maxes=np.zeros(11)
        self.mins=np.zeros(11)
        for row in x:
            for n, item in enumerate(row):
                if float(item) > self.maxes[n]:
                    self.maxes[n] = float(item)
                if float(item) < self.mins[n]:
                    self.mins[n] = float(item)
        for row in x:
            new_row = []
            for n, item in enumerate(row):
                new_row.append((float(item) - self.mins[n]) / self.maxes[n])
            x_last.append(new_row)
        x_last = np.array(x_last).astype(float)
        return x_last,y_last



    def train(self):
        x_last,y_last = self.preprocess(self.data)
        data_rows = x_last.shape[0]
        train_rate = int(0.8 * data_rows)
        x_train = x_last[:train_rate, :]
        x_test = x_last[train_rate:, :]
        y_train = y_last[:train_rate].astype(float)
        y_test = y_last[train_rate:].astype(float)
        self.model = Sequential(
            [
                layers.Dense(15, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(
            x_train,
            y_train,
            epochs=10,
            validation_data=(x_test, y_test),
        )
    def predict(self,data,have_y= False):
        if not have_y:
            zeros = np.zeros((data.shape[0],1))
            data = np.hstack((data, zeros))
        x,y = self.preprocess(data)
        return self.model.predict(x)

    @staticmethod
    def read_csv(file_path):
        file = open(file_path, 'r')
        csv_reader = csv.reader(file, delimiter=',')
        data = []
        for row in csv_reader:
            data.append(row)

        data = np.array(data)
        return data



breakpoint()

