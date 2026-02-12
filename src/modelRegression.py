import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from loader_data import load_csv

# PREPARAÇÃO DOS DADOS (Limpeza e Engenharia)
def preparar_dados(df):
    print("--- 1. Preparando Dados ---")
    
    # [OPCIONAL] Remoção baseada no PDF (Descomente se quiser usar)
    # if 'potencia_total_equipamentos' in df.columns:
    #    df = df.drop('potencia_total_equipamentos', axis=1)

    # Engenharia de Atributos
    df['densidade_habitacional'] = df['num_moradores'] / df['area_m2']

    # Remoção de Outliers (IQR)
    Q1 = df['consumo_energia'].quantile(0.25)
    Q3 = df['consumo_energia'].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    df_clean = df[(df['consumo_energia'] >= limite_inferior) & 
                  (df['consumo_energia'] <= limite_superior)].copy()
    
    print(f"Registros originais: {len(df)} -> Após limpeza: {len(df_clean)}")

    # Separação X e y
    X = df_clean.drop('consumo_energia', axis=1)
    y = df_clean['consumo_energia']
    
    return X, y


# TREINAMENTO (Escalonamento + MLP)
def treinar_modelo(X_train, y_train):
    print("\n--- 2. Treinando Modelo MLP ---")
    
    # Escalonamento (Importante: Fit apenas no treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Configuração da MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,              # Leve regularização L2 
        learning_rate='adaptive',  # Adapta a taxa se o erro estagnar 
        learning_rate_init=0.005,  # Começa com passos menores para precisão 
        max_iter=5000,             # Garante convergência
        random_state=42,
        early_stopping=True,       # Para se não melhorar (evita overfitting) 
        n_iter_no_change=20
    )

    mlp.fit(X_train_scaled, y_train)
    
    return mlp, scaler

# ==============================================================================
# 3. AVALIAÇÃO (Métricas)
# ==============================================================================
def avaliar_modelo(modelo, scaler, X_test, y_test):
    # Aplica o scaler treinado nos dados de teste
    X_test_scaled = scaler.transform(X_test)
    
    # Previsão
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("-" * 40)
    print(f"R² (Coeficiente de Determinação): {r2:.5f}")
    print(f"MAE (Erro Médio Absoluto): {mae:.2f} kWh")
    print(f"Iterações realizadas: {modelo.n_iter_}")
    print("-" * 40)
    
    return y_pred, r2

# ==============================================================================
# EXECUÇÃO DO SCRIPT
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. Carregar
        df_raw = load_csv()
        
        # 2. Preparar
        X, y = preparar_dados(df_raw)
        
        # 3. Dividir
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Treinar (Retorna o modelo e o scaler usado)
        modelo_mlp, scaler_treinado = treinar_modelo(X_train, y_train)
        
        # 5. Avaliar
        y_pred, r2_final = avaliar_modelo(modelo_mlp, scaler_treinado, X_test, y_test)
        

    except Exception as e:
        print(f"Ocorreu um erro crítico na execução: {e}")