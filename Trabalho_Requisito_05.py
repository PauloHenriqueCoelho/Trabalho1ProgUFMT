import numpy as np #Importa a biblioteca numpy
import pandas as pd #Importa a biblioteca pandas
import seaborn as sns #Biblioteca seaborn, ela faz o grafico
import matplotlib.pyplot as plt #Biblioteca matplotlib, ela serve para o seaborn funcionar e com ela podemos manipular mais informacoes do grafico
import sklearn #Importa sklearn
from sklearn.datasets import fetch_california_housing #Importa os dados que vamos trabalhar do sklearn

def main():

    # -------------------- Recebimento dos dados com o Pandas -------------------- #

    #Especifico que os dados importados agora estão na variavel casa_california
    casa_california = fetch_california_housing()

    #Uma matriz, pega os valores de casa_california. Pega todas as linhas, e cria 10 colunas para cada linha
    X = casa_california.data[:, 0:6]

    #Aqui especificamos as informações que estarão impressas na primeira coluna, no caso as informacoes de cada casa
    casa_informacao = ['MedInc','HousAge','AveRooms','AveBedrms','Population','AveOccup']

    #Criamos o DataFrame usando o panda, e especificamos que na primeira coluna deve ser preenchido com a nossa matriz casa_informacao
    df = pd.DataFrame(X, columns=casa_informacao)

    matriz_correlacao_spearman = df.corr(method='spearman')

    # -------------------- Geraçao de grafico -------------------- #

    # Cria uma figura de 10 por 8 (em polegadas)
    plt.figure(figsize=(10, 8))

    # Criar o heatmap da matriz de correlação de Pearson
    #annot exibe os valores dos coeficientes de correlação no mapa de calor.
    #cmap='coolwarm': define o esquema de cores. Cores "quentes" indicam correlação positiva, enquanto cores "frias" indicam correlação negativa.
    #fmt=".2f": Exibe os valores em duas casas decimais
    sns.heatmap(matriz_correlacao_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    # Titulo
    plt.title('Mapa de Calor da Correlação de Pearson')

    # Exibe o gráfico
    plt.show()


if __name__ == "__main__":
    main()

