import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carregar o arquivo CSV usando pandas
data = pd.read_csv('dados_empresas.csv')

# Converter a coluna 'data' para o tipo 'datetime'
data['data'] = pd.to_datetime(data['data'])

# Preencher valores ausentes com a média da coluna
data.fillna(data.mean(), inplace=True)

# Análise Exploratória de Dados

# Estatísticas Descritivas
print(data.describe())

# Correlação entre as variáveis
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlação entre as Variáveis')
plt.show()

# Gráficos

# Tendência do Lucro Líquido ao longo do tempo
plt.figure(figsize=(12, 6))
sns.lineplot(x='data', y='lucro liquido', hue='nome', data=data)
plt.title('Tendência do Lucro Líquido ao longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Lucro Líquido')
plt.xticks(rotation=45)
plt.show()

# Faturamento vs. Lucro Líquido
plt.figure(figsize=(8, 6))
sns.scatterplot(x='faturamento', y='lucro liquido', hue='nome', data=data)
plt.title('Faturamento vs. Lucro Líquido')
plt.xlabel('Faturamento')
plt.ylabel('Lucro Líquido')
plt.show()

# Previsões de Faturamento para cada empresa
for empresa, df_empresa in data.groupby('nome'):
    figs = []

    # Boxplot do Lucro Líquido
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y='lucro liquido', data=df_empresa)
    ax.set_title('Distribuição do Lucro Líquido - Empresa: ' + empresa)
    ax.set_ylabel('Lucro Líquido')
    figs.append(fig)

    # Histograma do Faturamento
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df_empresa['faturamento'], bins=10, kde=True)
    ax.set_title('Histograma do Faturamento - Empresa: ' + empresa)
    ax.set_xlabel('Faturamento')
    ax.set_ylabel('Frequência')
    figs.append(fig)

    # Lucro Líquido ao longo do tempo
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='data', y='lucro liquido', data=df_empresa)
    ax.set_title('Lucro Líquido ao longo do Tempo - Empresa: ' + empresa)
    ax.set_xlabel('Data')
    ax.set_ylabel('Lucro Líquido')
    ax.set_xticks(ax.get_xticks()[::len(df_empresa['data'])//10])
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    figs.append(fig)

    # Previsões de Faturamento
    train_data, test_data, train_target, test_target = train_test_split(df_empresa[['data']], df_empresa['faturamento'], test_size=0.2, random_state=1)
    train_data_numeric = train_data['data'].apply(lambda x: x.toordinal())
    test_data_numeric = test_data['data'].apply(lambda x: x.toordinal())
    model = LinearRegression()
    model.fit(train_data_numeric.values.reshape(-1, 1), train_target)
    predictions = model.predict(test_data_numeric.values.reshape(-1, 1))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(test_data['data'], test_target, label='Valores Reais')
    ax.plot(test_data['data'], predictions, label='Previsões')
    ax.set_xlabel('Data')
    ax.set_ylabel('Faturamento')
    ax.set_title('Previsões de Faturamento - Empresa: ' + empresa)
    ax.legend()
    figs.append(fig)

    # Mostrar os gráficos para a empresa atual
    for fig in figs:
        fig.tight_layout()
        plt.show()
