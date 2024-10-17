import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from src.config import Config

def treinar_modelo():

    dados = pd.read_csv(Config.TRAINING_DATA)
    X = dados.drop(columns=['classe'])
    y = dados['classe']

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    dados_teste = pd.read_csv(Config.TEST_DATA)
    X_teste = dados_teste.drop(columns=['classe'])
    y_teste = dados_teste['classe']
    predicoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, predicoes)

    print(f"Acurácia no conjunto de teste: {acuracia:.2f}")

    joblib.dump(modelo, f"{Config.MODEL_DIR}/modelo_treinado.pkl")
    print("Modelo treinado e salvo.")