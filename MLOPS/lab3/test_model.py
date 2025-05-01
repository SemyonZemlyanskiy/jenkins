import pickle
import numpy as np

# Загрузка модели
with open('mlruns/326404350029200275/4d7692e320384c7490be20b1e6193674/artifacts/model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Пример входных данных (замени на свои)
X_new = np.array([[7200, 2003.0, 200000.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   2.0, 3.0, 1.0, 1.0, 2.0]])  # shape (1, 28)

# Предсказание
y_pred = model.predict(X_new)

print("Предсказание:", y_pred)