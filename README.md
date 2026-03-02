# Neural Network — Digit Recognition

Rede neural do zero para reconhecimento de dígitos manuscritos (0–9), com interface web interativa e pesos pré-treinados.

## Visão Geral

| Componente | Descrição |
|---|---|
| `main.py` | Treinamento interativo em Python com visualização e testes |
| `export_weights.py` | Treina a rede e exporta os pesos para `weights.json` / `weights.js` |
| `index.html` | Interface web: desenhe um dígito e veja o resultado em tempo real |
| `verify.py` | Verifica os pesos exportados usando Python |
| `verify.js` | Verifica os pesos exportados usando Node.js |
| `draw_table.pde` | Sketch Processing para coleta de dígitos desenhados à mão |

## Arquitetura da Rede

```
Entrada (784)  →  Camada Oculta (200)  →  Saída (10)
  28×28 px        sigmoid                  dígitos 0–9
```

- **Função de ativação:** Sigmoid
- **Learning rate:** 0.1
- **Epochs:** 7 (treinamento Python) / 5 (re-treinamento no browser)
- **Augmentação:** rotação de +10° e −10° para cada amostra
- **Dados:** 1 000 amostras de treino (MNIST subset)

## Bugs Corrigidos

| Bug | Antes | Depois |
|---|---|---|
| `train_file.readlines()` dentro do loop de epochs | A lista ficava vazia após a 1ª epoch → treinava só 1 vez | `readlines()` chamado **antes** do loop; todas as epochs percorrem os dados |
| Learning rate muito baixo | `0.01` | `0.1` |
| Augmentação falsa no browser | Os mesmos dados repetidos 3× | Rotações reais (+10°, −10°) via transformação de canvas |
| Pesos aleatórios na interface web | A rede iniciava sem treino → sempre previa o mesmo dígito | `weights.js` carregado automaticamente ao abrir `index.html` |

## Como Usar

### 1. Instalar dependências Python

```bash
pip install numpy scipy matplotlib
```

### 2. Gerar pesos pré-treinados

```bash
python export_weights.py
```

Isso treina a rede com `train-1000.csv` (7 epochs, augmentação ±10°) e exporta `weights.json` e `weights.js`.

### 3. Interface Web

Abra `index.html` no browser. Os pesos são carregados automaticamente do `weights.js`.

- **Desenhe** um dígito no canvas (0–9)
- **Brush size:** ajuste o tamanho do pincel com o slider
- **↩ Undo:** desfaz o último traço
- **Classify:** a rede exibe o dígito reconhecido
- **Re-train:** re-treina a rede no browser com `train-1000.csv`

### 4. Treinamento e testes interativos (Python)

```bash
python main.py
```

Menu de opções:
1. Testar um dígito da folha de testes (`test-10.csv`)
2. Medir desempenho geral
3. Desenhar e reconhecer com `draw_table`

### 5. Verificar pesos exportados

```bash
python verify.py        # requer weights.json e test-10.csv
node verify.js          # requer weights.json e test-10.csv
```

## Estrutura de Arquivos

```
├── main.py              # Script de treino e testes interativos
├── export_weights.py    # Exporta pesos para uso na web
├── index.html           # Interface web (sem servidor necessário)
├── verify.py            # Verificação dos pesos em Python
├── verify.js            # Verificação dos pesos em Node.js
├── draw_table.pde       # Sketch Processing para coleta de dados
├── weights.json         # Pesos pré-treinados (gerado pelo export_weights.py)
├── weights.js           # Pesos em JS (carregado pelo index.html)
├── train-1000.csv       # 1 000 amostras de treino (MNIST)
├── train-100.csv        # 100 amostras de treino
├── test-10.csv          # 10 amostras de teste
└── number.csv           # Dígito desenhado via draw_table
```

## Detalhes da Implementação

### Backpropagation

Os pesos **não** são transpostos em relação à formulação clássica do Tariq:

```
ΔWih = lr × (inputsᵀ · (hidden_errors × σ(H) × (1 − σ(H))))
ΔWho = lr × (hidden_outputsᵀ · (output_errors × σ(O) × (1 − σ(O))))
```

Onde: `lr` = learning rate, `ᵀ` = transposta, `σ` = sigmoid, `H` = saídas da camada oculta, `O` = saídas da camada final, `×` = multiplicação element-wise, `·` = produto matricial.

### Inicialização dos pesos

```
Wih ~ N(0, 1/√input_nodes)   shape: (784, 200)
Who ~ N(0, 1/√hidden_nodes)  shape: (200,  10)
```
