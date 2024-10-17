# -----------------------------------
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# -----------------------------------
import pandas as pd
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from prettytable import PrettyTable

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings('ignore')

# Configurações
from tree_classification.settings import get_path_to_data

# Ler o arquivo CSV da pasta de dados
FILE_INPUT = 'covtype.csv'
input_data = pd.read_csv(f'{get_path_to_data()}/{FILE_INPUT}')

# -----------------------------------
# 2. CARREGAMENTO E VISUALIZAÇÃO DOS DADOS
# -----------------------------------
print("\n🌳--- 2. CARREGAMENTO E VISUALIZAÇÃO DOS DADOS ---🌳")
print("\n--- Primeiras linhas do DataFrame ---")
print(input_data.head())

print("\n--- Informações sobre o DataFrame ---")
print(input_data.info())

print("\n--- Estatísticas descritivas ---")
print(input_data.describe())

print("\n--- Valores ausentes por coluna ---")
print(input_data.isnull().sum())

print("\n--- Valores únicos em 'Cover_Type' ---")
print(f"Valores únicos em 'Cover_Type': {input_data['Cover_Type'].unique()}")

# -----------------------------------
# 3. TRANSFORMAÇÃO DOS DADOS
# -----------------------------------
print("\n🔄--- 3. TRANSFORMAÇÃO DOS DADOS ---🔄")
# Certificando que 'Cover_Type' é do tipo int
input_data['Cover_Type'] = input_data['Cover_Type'].astype(int)

# Dividir variáveis independentes (X) e dependentes (y)
X = input_data.drop('Cover_Type', axis=1)  # Usando 'Cover_Type' como a coluna alvo
y = input_data['Cover_Type']  # Usando 'Cover_Type' como a coluna alvo

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificando tipos e valores de y_train
print("\n🔍--- Informações sobre y_train ---")
print(f"Tipo de y_train: {y_train.dtype}")
print(f"Valores únicos em y_train: {y_train.unique()}")

# -----------------------------------
# 4. AJUSTE DO MODELO DE ÁRVORE DE DECISÃO
# -----------------------------------
print("\n🌲--- 4. AJUSTE DO MODELO DE ÁRVORE DE DECISÃO ---🌲")
# Medir o tempo de treinamento
start_time = time.time()

# Criar e ajustar o modelo
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)  # Ajusta o modelo aos dados de treinamento
training_time = time.time() - start_time

# Prever resultados
y_pred = dt_classifier.predict(X_test)  # Previsão com o conjunto de teste

# Calcular o tempo de treinamento
training_time = time.time() - start_time
print(f"✅ Modelo treinado com sucesso em {training_time:.4f} segundos.")

# -----------------------------------
# 5. AVALIAÇÃO DO MODELO
# -----------------------------------
print("\n📊--- 5. AVALIAÇÃO DO MODELO ---📊")
# Criar e exibir a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Usando PrettyTable para a matriz de confusão
print("\n--- Matriz de Confusão ---")
conf_table = PrettyTable()
conf_table.field_names = ["Predicted \\ Actual"] + [str(i) for i in range(1, 8)]  # Ajuste para suas classes

# Adicionando linhas à tabela
for i, row in enumerate(conf_matrix):
    conf_table.add_row([str(i + 1)] + row.tolist())  # Adiciona a linha correspondente ao índice da classe

print(conf_table)  # Exibe a tabela da matriz de confusão

# Relatório de classificação
print("\n--- Relatório de Classificação ---")
report = classification_report(y_test, y_pred, output_dict=True)  # Obtém o relatório em formato de dicionário

# Usando PrettyTable para o relatório de classificação
report_table = PrettyTable()
report_table.field_names = ["Classe", "Precisão", "Recall", "F1-Score", "Suporte"]

# Adicionando dados ao relatório
for class_label, metrics in report.items():
    if isinstance(metrics, dict):  # Ignorar as métricas médias
        report_table.add_row([class_label, f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}", f"{metrics['f1-score']:.2f}", metrics['support']])

print(report_table)  # Exibe o relatório de classificação

# Informações adicionais sobre o desempenho do modelo
print("\n📈--- DESEMPENHO DO MODELO ---📈")
performance_table = PrettyTable()
performance_table.field_names = ["Métrica", "Valor"]
performance_table.add_row(["Tempo de Treinamento", f"{training_time:.4f} segundos"])
performance_table.add_row(["Número de Nós", dt_classifier.tree_.node_count])
performance_table.add_row(["Profundidade da Árvore", dt_classifier.tree_.max_depth])

print(performance_table)  # Exibe informações adicionais do modelo
