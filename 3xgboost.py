# XGBoost

# Загружаем библиотеки
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загружаем данные
dataset = pd.read_csv('1dataset.csv')
training_set_scaled = dataset.iloc[:, 4:5].values

# Разворачиваем данные
X_train = []
y_train = []
for i in range(60, 5717): #Всего 5727 строк. Трейнинг - последние 10
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Запускаем Берлагу (c) Ильф и Петров
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)

# Готовимся прогнозировать прогнозы
real_price = dataset.iloc[5717:5728, 4:5].values #10
inputs = dataset.iloc[5657:5728, 4:5].values #10+60 #inputs = inputs.reshape(-1,1)
X_test = []
for i in range(61, 71):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

# Прогнозируем прогноз!
predicted_price = regressor.predict(X_test)

# Сохраняем в эксельку ... потом повертим в Экселе или в Tableau или ещё где
df_real_price = pd.DataFrame(inputs)
df_real_price.to_excel("31real.xlsx")
df_predicted_price = pd.DataFrame(predicted_price)
df_predicted_price.to_excel("32prediction.xlsx")

# Нарисуем график
plt.plot(real_price, color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('XGBoost Prediction')
plt.xlabel('Time')
plt.ylabel('Real Price')
plt.legend()
plt.show()

# Оценим качество прогноза через RMSE (Root Mean Squared Error)
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_price, predicted_price))