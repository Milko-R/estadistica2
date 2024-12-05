import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

# Cargar los datos
data = pd.read_csv('Base_financiera_1.csv', delimiter=';')

# Imprimir las columnas del DataFrame
print(data.columns)

# Convertir variables categóricas a variables dummy
data = pd.get_dummies(data)

# Separar las características (X) y la variable objetivo (y)
X = data.drop('Ingreso', axis=1)
y = data['Ingreso']

# Dividir el conjunto de datos en entrenamiento y prueba con proporción 70% y 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir los modelos a probar
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor()
}

# Definir los hiperparámetros a optimizar para cada modelo
param_distributions = {
    'RandomForest': {
        'n_estimators': randint(100, 1000),
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    },
    'GradientBoosting': {
        'n_estimators': randint(100, 200),
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
}

# Probar cada modelo
for name, model in models.items():
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions[name], n_iter=50, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Imprimir los resultados
    print(f'{name} Model:')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Best Parameters: {random_search.best_params_}\n')
