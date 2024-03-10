from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import argparse

import keras
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Bidirectional, BatchNormalization, Embedding

def get_training_data(df):
    train_dates = pd.to_datetime(df['Date'])
    cols = ['Close']

    df_for_training = df[cols].astype(float)

    return train_dates, df_for_training


def scale_dataset(df_for_training):
    scaler = StandardScaler()
    
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    
    return df_for_training_scaled


def get_scaler(df_for_training):
    scaler = StandardScaler()
    
    scaler = scaler.fit(df_for_training)
    return scaler
    


def get_training_sets(df, n_future=1,n_past=100):
    train_dates, df_for_training = get_training_data(df)
    df_for_training_scaled = scale_dataset(df_for_training)
    
    trainX=[]
    trainY=[]

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i-n_past:i,0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i+n_future-1:i+n_future,0])

    trainX,trainY=np.array(trainX), np.array(trainY)

    return trainX, trainY


def main():
    parser = argparse.ArgumentParser(description='ML model training CLI')
    parser.add_argument('--symbol', required=True, type=str, help='Stock symbol')
    parser.add_argument('--period1', required=True, type=int, help='Start timestamp (Unix)')
    parser.add_argument('--period2', required=True, type=int, help='End timestamp (Unix)')
    parser.add_argument('--interval', required=True, type=str, help='Interval')

    args = parser.parse_args()

    symbol = args.symbol
    period1 = args.period1
    period2 = args.period2
    interval = args.interval

    csvFile = f'data/csv/{symbol}_{period1}_{period2}_{interval}.csv'
    
    if not os.path.exists(csvFile):
        print(f'file does not exist for symbol={symbol}')
        exit() 
    else:
        modelFilePath = f'data/models/model.{symbol}.h5'

        if os.path.exists(modelFilePath):
            print(f'model already exists for symbol={symbol}')
            exit()

        # Preping Stock Data 
        df = pd.read_csv(csvFile)
        train_dates, df_for_training = get_training_data(df)
        trainX, trainY = get_training_sets(df)

        print('trainX shape==={}.'.format(trainX.shape))
        print('trainY shape==={}.'.format(trainY.shape))

        
        # Building the model from historical stock data
        model=Sequential()
        model.add(Bidirectional(LSTM(32,activation='relu',return_sequences=True),input_shape=(trainX.shape[1],trainX.shape[2])))
        model.add(Dropout(0.2))

        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.2))

        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer='adam',loss='mse')

        callback = [
            keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
            keras.callbacks.ModelCheckpoint(filepath=modelFilePath, monitor='val_loss',save_best_only=False),
            keras.callbacks.TensorBoard(log_dir='./logs')
        ]

        model.fit(trainX,trainY,epochs=40,batch_size=16,validation_split=0.1,verbose=1,callbacks=callback)
        model.save(modelFilePath)
        del model
        print(f'{symbol} model saved')
        


if __name__ == '__main__':
    main()



