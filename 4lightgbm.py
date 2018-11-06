# LightGBM

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

# И сразу готовимся прогнозировать прогнозы
real_price = dataset.iloc[5717:5728, 4:5].values #10
inputs = dataset.iloc[5657:5728, 4:5].values #10+60 #inputs = inputs.reshape(-1,1)
X_test = []
y_test = []
for i in range(61, 71):
    X_test.append(inputs[i-60:i, 0])
    y_test.append(training_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)


# Запускаем Берлагу (c) Ильф и Петров
import lightgbm as lgb

# Специальным образом создаём датасеты для lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Задаём параметры в виде словаря
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 131,
    'learning_rate': 1.2,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# тренируем-с
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)


# Прогнозируем прогноз!
predicted_price = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# Сохраняем в эксельку ... потом повертим в Экселе или в Tableau или ещё где
df_real_price = pd.DataFrame(inputs)
df_real_price.to_excel("41real.xlsx")
df_predicted_price = pd.DataFrame(predicted_price)
df_predicted_price.to_excel("42prediction.xlsx")

# Нарисуем график
plt.plot(real_price, color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('LightGBM Prediction')
plt.xlabel('Time')
plt.ylabel('Real Price')
plt.legend()
plt.show()

# Оценим качество прогноза через RMSE (Root Mean Squared Error)
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_price, predicted_price))