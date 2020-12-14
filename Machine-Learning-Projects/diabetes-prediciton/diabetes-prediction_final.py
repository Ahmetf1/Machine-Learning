import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping

class Pipeline():
    def __init__(self,data):
        self.data=data
    def preprocess(self):
        self.data = np.array(self.data)
        self.data = self.data[1:, :]

        x = self.data[:, :8].astype(float)
        y = self.data[:, 8].astype(float)

        return x, y

    def train(self,show_graph=False,save_model=False,model_name="model.h5"):
        x,y = self.preprocess(self.data)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

        scaler = MinMaxScaler()
        scaler.fit(train_x)
        scaler.transform(train_x)
        scaler.transform(test_x)

        model = Sequential()
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
        stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
        model.fit(train_x, train_y, verbose=1, batch_size=64, callbacks=[stop], epochs=700,
                  validation_data=(test_x, test_y))
        if show_graph:
            import pandas as pd
            graph = pd.DataFrame(model.history.history)
            graph.plot()
        if save_model:
            model.save(model_name)

        @staticmethod
        def read_csv(file_path):
            data = []
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    data.append(row)
            return data

