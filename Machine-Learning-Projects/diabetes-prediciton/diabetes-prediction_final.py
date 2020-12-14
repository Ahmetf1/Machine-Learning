import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle


class Pipeline:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        self.data = np.array(self.data)
        self.data = self.data[1:, :]

        x = self.data[:, :8].astype(float)
        y = self.data[:, 8].astype(float)

        return x, y

    def train(self, save_model=False, model_name="model.h5"):
        x, y = self.preprocess()
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

        self.scaler = MinMaxScaler()
        self.scaler.fit(train_x)
        train_x = self.scaler.transform(train_x)
        test_x = self.scaler.transform(test_x)

        self.model = Sequential()
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(8, activation="relu"))
        self.model.add(Dense(8, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
        stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
        self.model.fit(train_x, train_y, verbose=1, batch_size=64, callbacks=[stop], epochs=10,
                       validation_data=(test_x, test_y))

        if save_model:
            self.model.save(model_name)

    def predict(self, data_predict):
        data_predict=np.array(data_predict)
        x = data_predict[1:, :8].astype(float)
        result=self.model.predict(x)
        return result

    @staticmethod
    def read_csv(file_path):
        data_x = []
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data_x.append(row)
        return data_x

    def save(self, file_path):
        pickle.dump(self, open(file_path, 'wb'))

    @staticmethod
    def load(file_path):
        return pickle.load(open(file_path, 'rb'))


pipe = Pipeline(Pipeline.read_csv(r"...\diabetes.csv"))
pipe.train()
print(pipe.predict(Pipeline.read_csv(r"...\diabetes.csv")))
