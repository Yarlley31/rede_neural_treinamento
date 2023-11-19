import numpy as np

# Criando a função de ativação (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inicialização dos pesos e bias
np.random.seed(0)
input_size = 2
hidden_size1 = 2
hidden_size2 = 2
output_size = 1

# criado peso e bia da camada oculta 1
peso1 = np.random.randn(input_size, hidden_size1)
bia1 = np.zeros((1, hidden_size1))

# criando peso e bia da camada oculta 2
peso2 = np.random.randn(hidden_size1, hidden_size2)
bia2 = np.zeros((1, hidden_size2))

# Criando peso e bia da camada de saida
peso3 = np.random.randn(hidden_size2, output_size)
bia3 = np.zeros((1, output_size))

# Entrada
input_data = np.array([[10, 10]])

# Feedforward
layer1_output = sigmoid(np.dot(input_data, peso1) + bia1)
layer2_output = sigmoid(np.dot(layer1_output, peso2) + bia2)

output = sigmoid(np.dot(layer2_output, peso3) + bia3)

print("-"*38)
print("Entrada de rede neural:", input_data)
print("Saída de rede neural:", output)
print("-"*38)