import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config

def preprocessar_dados():

    dados = pd.read_csv(f"{Config.DATA_DIR}/raw/dados.csv")
    
    dados = dados.drop(columns=['coluna_irrelevante'])
    dados['valor_normalizado'] = dados['valor'] / dados['valor'].max()

    treino, teste = train_test_split(dados, test_size=0.2, random_state=42)

    treino.to_csv(Config.TRAINING_DATA, index=False)
    teste.to_csv(Config.TEST_DATA, index=False)

    print("Dados pré-processados e salvos.")