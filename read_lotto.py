import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow import keras
from keras.optimizers import Adam

df = pd.read_csv('storico01-oggi.txt', sep='\t', header=None, na_filter=False)
df.drop([0], axis=1, inplace=True)
# create a set of strings using the first column
cities = set(df[1].astype(str))

cities_df = {}

for city in cities:
    # select the rows where the city column is equal to the current city
    subset = df[df[1] == city]
    # print the first 10 rows of the subset
    trunc_sub = df.drop([1], axis=1)
    cities_df[city] = trunc_sub
    # print(subset.head(10))
    scaler = StandardScaler().fit(cities_df[city])
    transformed_dataset = scaler.transform(cities_df[city])
    transformed_df = pd.DataFrame(data=transformed_dataset, index=cities_df[city].index)
    print(transformed_df.head())
    number_of_rows = cities_df[city].values.shape[0]
    window_length = 7
    number_of_features = cities_df[city].values.shape[1]
    
    X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
    y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)
    for i in range(0, number_of_rows-window_length):
        X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
        y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]
    print(X.shape)
    print(y.shape)
    # import Sequential from keras
    model = Sequential()
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = False)))
    model.add(Dense(59))
    model.add(Dense(number_of_features))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])
    model.fit(x=X, y=y, batch_size=100, epochs=300, verbose=2)
    to_predict = df.tail(8)
    print(f"{city} -> to_predict {to_predict}")
# head = cities_df["NA"].iloc[::100000, :]

# cities_df["NA"].head(1000).plot(x=1,y=2, style='o', figsize=(20, 18)).figure.savefig('plot.png')
print("OK")