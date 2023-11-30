# Importações do código
import math
from random import uniform as unf

# Função de ativação
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

# Derivada da função de ativação sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Sorteia números de pesos
def random_pesos(pesos):
    for x in range(0, 4):
        for y in range(0, 2):
            pesos[x][y] = unf(-1, 1)

    return pesos

# Sorteia números de Bias
def random_bias(bias):
    for x in range(0, 5):
        bias[x] = unf(-10, 10)

    return bias

# Função de erro
def função_perda(x, y):
    perda = x - y
    if perda < 0.0:
        perda = perda * -1

    return perda

# Taxa de aprendizado
taxa_apredizado = 0.1

# Define neuronios
entrada = [10, 10]
camada_oculta_1 = [0, 0]
camada_oculta_2 = [0, 0]
camada_de_saida = 0

# Define pesos e bias
pesos = [[0, 0], [0, 0], [0, 0], [0, 0]]
bias = [0, 0, 0, 0, 0]

# Define retorno esperado
resultado_esperado = 0.5

print('start: ')

# Loop que termina quando acha números de bias e pesos que retorna aproximadamente igual ao resultado esperado
for época in range(10000):  # Número máximo de iterações para evitar loops infinitos
    bias = random_bias(bias)
    pesos = random_pesos(pesos)

    # Forward pass
    camada_oculta_1[0] = sigmoid((entrada[0] * pesos[0][0]) + bias[0])
    camada_oculta_1[1] = sigmoid((entrada[1] * pesos[0][1]) + bias[1])

    camada_oculta_2[0] = sigmoid(((camada_oculta_1[0] * pesos[1][0]) + (camada_oculta_1[1] * pesos[1][1])) + bias[2])
    camada_oculta_2[1] = sigmoid(((camada_oculta_1[0] * pesos[2][0]) + (camada_oculta_1[1] * pesos[2][1])) + bias[3])

    camada_de_saida = sigmoid(((camada_oculta_2[0] * pesos[3][0]) + (camada_oculta_2[1] * pesos[3][1])) + bias[4])

    # Calcula o erro
    perda = função_perda(resultado_esperado, camada_de_saida)

    # Backward pass (backpropagation)
    # Atualiza pesos e bias para reduzir o erro
    erro_saida = resultado_esperado - camada_de_saida
    delta_saida = erro_saida * sigmoid_derivative(camada_de_saida)

    erro_oculta_2 = delta_saida * pesos[3][0] + delta_saida * pesos[3][1]
    delta_oculta_2 = erro_oculta_2 * sigmoid_derivative(camada_oculta_2[0])

    erro_oculta_1 = delta_oculta_2 * pesos[1][0] + delta_oculta_2 * pesos[2][0]
    delta_oculta_1 = erro_oculta_1 * sigmoid_derivative(camada_oculta_1[0])

    # Atualiza pesos e bias
    pesos[3][0] += taxa_apredizado * delta_saida * camada_oculta_2[0]
    pesos[3][1] += taxa_apredizado * delta_saida * camada_oculta_2[1]

    pesos[1][0] += taxa_apredizado * delta_oculta_2 * camada_oculta_1[0]
    pesos[2][0] += taxa_apredizado * delta_oculta_2 * camada_oculta_1[0]

    pesos[0][0] += taxa_apredizado * delta_oculta_1 * entrada[0]
    pesos[0][1] += taxa_apredizado * delta_oculta_1 * entrada[1]

    bias[4] += taxa_apredizado * delta_saida
    bias[2] += taxa_apredizado * delta_oculta_2
    bias[3] += taxa_apredizado * delta_oculta_2
    bias[0] += taxa_apredizado * delta_oculta_1
    bias[1] += taxa_apredizado * delta_oculta_1

    if época % 1000 == 0:
        print(f'época: {época}, perda: {perda}')

    # Verifica se o erro é pequeno o suficiente
    if perda < 0.0001:
        print('Treinamento concluído!')
        print('Entrada: ', entrada)
        print('Saída:', camada_de_saida)
        break
