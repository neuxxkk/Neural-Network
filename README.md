# Neural Network — Reconhecimento de Dígitos

Rede neural feita do zero em **Python** para reconhecer dígitos manuscritos (0–9), treinada com o dataset MNIST.

## Estrutura

| Arquivo | Tecnologia | Função |
|---|---|---|
| `main.py` | Python | Rede neural, treinamento e inferência |
| `draw_table.pde` | Processing (Java) | Interface gráfica para desenhar o dígito |

> **A inteligência fica toda no Python.** O Processing é usado apenas para abrir uma janela onde o usuário desenha o número; o resultado é salvo em `number.csv` e lido pelo Python para obter a predição.

## Dependências Python

```
numpy
scipy
matplotlib
```

Instale com:

```bash
pip install numpy scipy matplotlib
```

## Como usar

1. Execute o script principal:

```bash
python main.py
```

2. Aguarde o treinamento terminar e escolha uma opção:

```
1 - Um dígito da folha de testes
2 - Teste de desempenho
3 - Desenhe você mesmo
0 - Sair
```

3. Na opção **3**, a janela do Processing abre. Desenhe o dígito, clique em **Done** (verde) e o Python exibe o número reconhecido.

## Arquitetura da rede

- Entrada: 784 neurônios (imagem 28×28 pixels)
- Camada oculta: 200 neurônios
- Saída: 10 neurônios (dígitos 0–9)
- Ativação: Sigmoid
- Taxa de aprendizado: 0.01
- Data augmentation: rotações de ±10° por amostra
