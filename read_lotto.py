import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow import keras
from keras.optimizers import Adam


# import Sequential from keras

def rnn(units, dense_units, input_shape, number_of_features, X, y, epochs, batch_size, learning_rate):
    """
        Default settings used in medium article:
        units=240
        dense_units=59
        epochs=300
        batch_size=100
        learning_rate=0.0001
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(units, input_shape = input_shape, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units, input_shape = input_shape, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units, input_shape = input_shape, return_sequences = True)))
    model.add(Bidirectional(LSTM(units, input_shape = input_shape, return_sequences = False)))
    model.add(Dense(dense_units))
    model.add(Dense(number_of_features))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss ='mse', metrics=['accuracy'])
    model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, verbose=2)
    
    return model

def build_cities_dict(df: pd.DataFrame, cities: set) -> dict:
    """
    Build a dictionary of dataframes, one for each city
    """
    cities_dict = {}
    for city in cities:
        # select the rows where the city column is equal to the current city
        subset = df[df[1] == city]
        # remove the city column
        trunc_sub = df.drop([1], axis=1)
        cities_dict[city] = trunc_sub
    return cities_dict

def build_training_set(dataframe: pd.DataFrame):
    scaler = StandardScaler().fit(dataframe)
    transformed_dataset = scaler.transform(dataframe)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=dataframe.index)
    print(transformed_df.head())

    number_of_rows = dataframe.values.shape[0]
    window_length = 7
    # number of balls for a "ruota"
    number_of_features = dataframe.values.shape[1]

    # create the training set
    X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
    y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)
    for i in range(0, number_of_rows-window_length):
        X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
        y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]
    input_shape=(window_length, number_of_features)
    return X, y, input_shape, number_of_features

def main():
    print("Hello World!")
    df = pd.read_csv('storico.txt', sep='\t', header=None, na_filter=False)
    # drop the dates
    df.drop([0], axis=1, inplace=True)
    # create a set containing all the cities of Lotto, using the first column
    cities = set(df[1].astype(str))

    cities_dict = build_cities_dict(df, cities)

    # normalize the data into format [0,1] and build the training set
    X, y, input_shape, number_of_features = build_training_set(cities_dict["NA"])

    print(f"X shape: {X.shape}; y shape: {y.shape}")
    
    print("Training the model on a single city...")
    model = rnn(units=240, 
                dense_units=59, 
                input_shape=input_shape, 
                number_of_features=number_of_features, 
                X=X, 
                y=y, 
                epochs=1, 
                batch_size=100, 
                learning_rate=0.0001)

    print("Predicting the next lottery numbers...")

    city_df = cities_dict["NA"]

    to_predict = city_df.tail(8)
    to_predict.drop([to_predict.index[-1]],axis=0, inplace=True)


    to_predict = np.array(to_predict)
    scaled_to_predict = scaler.transform(to_predict)

    y_pred = model.predict(np.array([scaled_to_predict]))
    print("The predicted numbers in the last lottery game are:", scaler.inverse_transform(y_pred).astype(int)[0])

    prediction = city_df.tail(1)
    prediction = np.array(prediction)
    print("The actual numbers in the last lottery game were:", prediction[0])
    print("END")
if __name__ == "__main__":        
    main()
