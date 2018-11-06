#РННка

# Загружаем библиотеки
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загружаем данные
dataset = pd.read_csv('dataset.csv')
training_set = dataset.iloc[:, 4:5].values

# Нормализуем
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Разворачиваем данные для РННки
X_train = []
y_train = []
for i in range(60, 5717): #Всего 5727 строк. Трейнинг - последние 10
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Загружаем библиотеки Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Строим РННку
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

# Компилируем
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Учим РННку
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


## Выделяем тестовые данные из датасета - последние 10 + 60 для разворачивания
#dataset_test = pd.read_csv('dataset.csv')
#real_stock_price = dataset_test.iloc[5657:5728, 4:5].values
#
## Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train['<OPEN>'], dataset_test['<OPEN>']), axis = 0)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#inputs = inputs.reshape(-1,1)

# Выделяем тестовые данные из датасета - последние 10 отсчетов + 60 для разворачивания
real_stock_price = dataset.iloc[5717:5728, 4:5].values #10
inputs = dataset.iloc[5657:5728, 4:5].values #10+60
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 71):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
inputs1 = sc.inverse_transform(inputs)

# Сохраняем в эксельку ... потом повертим в Экселе или в Tableau или ещё где
df_predicted_stock_price = pd.DataFrame(predicted_stock_price)
df_predicted_stock_price.to_excel("df_predicted_stock_price.xlsx")
df_real_stock_price = pd.DataFrame(inputs)
df_real_stock_price.to_excel("df_real_stock_price.xlsx")

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#Evaluating the RNN - using RMSE (Root Mean Squared Error)
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))