<div align="center">

# 🧠 Neural Network — Digit Recognition

*Rede neural do zero, sem frameworks, capaz de ler sua caligrafia no navegador*

[![Language](https://img.shields.io/badge/Language-Python%203-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Web](https://img.shields.io/badge/Interface-HTML%2FJS-F7DF1E?style=flat-square&logo=javascript&logoColor=black)]()
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-00d4aa?style=flat-square)]()
[![Algorithm](https://img.shields.io/badge/Algoritmo-Backpropagation-00d4aa?style=flat-square)]()

</div>

---

Rede neural **MLP (Multilayer Perceptron)** implementada do zero em Python e JavaScript — sem PyTorch, sem TensorFlow — capaz de reconhecer dígitos manuscritos (0–9) desenhados diretamente no navegador.

Todo o cálculo algébrico (multiplicação de matrizes, derivadas, descida de gradiente) está exposto e documentado no código, tornando o projeto uma referência didática de como redes neurais realmente funcionam por dentro.

---

## 📐 Visão Geral

| Componente | Tecnologia | Responsabilidade |
|------------|-----------|-----------------|
| `main.py` | Python 3 | Treinamento interativo, backpropagation e testes pelo terminal |
| `export_weights.py` | Python 3 | Treina e exporta os pesos para `weights.json` / `weights.js` |
| `index.html` | HTML + JS | Interface web: desenhe um dígito e veja o resultado em tempo real |
| `verify.py` | Python 3 | Verifica os pesos exportados contra o dataset de teste |
| `verify.js` | Node.js | Verifica os pesos exportados em ambiente JS |
| `draw_table.pde` | Processing | Sketch para coleta de dígitos desenhados à mão |

---

## 🧬 Arquitetura da Rede

```
Entrada (784)  →  Camada Oculta (200)  →  Saída (10)
  28×28 px           sigmoid               dígitos 0–9
```

- **Camada de entrada:** cada pixel do canvas 28×28 vira um input contínuo normalizado (`0.01` a `0.99`)
- **Camada oculta:** extrai padrões primitivos (linhas, curvas, arestas) via sigmoid
- **Camada de saída:** cada nó gera a confiança (%) do desenho representar um dígito de 0 a 9

| Parâmetro | Valor |
|-----------|:-----:|
| Função de ativação | Sigmoid |
| Learning rate | `0.1` |
| Epochs (Python) | `7` |
| Epochs (browser) | `5` |
| Augmentação | Rotação ±10° por amostra |
| Dados de treino | 1.000 amostras (MNIST subset) |

### Inicialização dos Pesos

```
Wih ~ N(0, 1/√input_nodes)   shape: (784, 200)
Who ~ N(0, 1/√hidden_nodes)  shape: (200,  10)
```

---

## ✨ Funcionalidades e Otimizações

### Center of Mass Normalization

O maior vilão de acurácia em redes estáticas (sem camadas convolucionais) é a posição inconsistente do desenho humano. Para resolver isso, implementamos um **algoritmo de Centro de Massa Bounding-Box** em JS puro (`preProcessImage`):

1. Traça uma bounding box ao redor de todo o traço do usuário
2. Reduz essa região para **20×20 px** via interpolação bilinear
3. Reposiciona o conteúdo nas coordenadas centrais do frame 28×28

> Isso reflete exatamente como o dataset MNIST foi pré-processado — o que elimina a maior fonte de erro em inferência front-end.

Um **Live Preview 28×28** no canto da tela mostra em tempo real o que a rede realmente recebe como input.

---

### Pesos sem Web Server (contorno de CORS)

Carregar arquivos `.json` localmente via `file:///` dispara bloqueios de CORS no browser. A solução: o `export_weights.py` salva as matrizes diretamente como **JavaScript assíncrono interpretável** (`weights.js`), carregado automaticamente pelo `index.html` sem precisar de localhost ou servidor.

> Resultado: qualquer pessoa abre o projeto com um duplo clique no HTML, sem configuração.

---

### Data Augmentation

Para cada amostra de treino, a rede é treinada **três vezes**:

1. Imagem original
2. Imagem rotacionada **+10°** (via `scipy.ndimage`)
3. Imagem rotacionada **−10°** (via `scipy.ndimage`)

Isso triplica artificialmente o conjunto de treino e produz pesos mais robustos e generalizáveis.

---

## 🐛 Bugs Corrigidos

| Bug | Antes | Depois |
|-----|-------|--------|
| `readlines()` dentro do loop de epochs | Lista ficava vazia após a 1ª epoch — treinava só 1 vez | Chamado **antes** do loop; todas as epochs percorrem os dados |
| Learning rate muito baixo | `0.01` | `0.1` |
| Augmentação falsa no browser | Os mesmos dados repetidos 3× | Rotações reais (±10°) via transformação de canvas |
| Pesos aleatórios na interface web | Rede sem treino → sempre previa o mesmo dígito | `weights.js` carregado automaticamente ao abrir `index.html` |

---

## 🚀 Como Usar

### Interface Web (uso padrão)

Plug-and-play — basta ter o `weights.js` gerado na mesma pasta:

1. Abra `index.html` com duplo clique no explorador de arquivos
2. Desenhe um dígito (0–9) no canvas

| Controle | Ação |
|:--------:|------|
| Canvas | Desenhe com o mouse |
| Slider | Ajuste o tamanho do pincel |
| **↩ Undo** | Desfaz o último traço |
| **Classify** | Classifica o dígito desenhado |
| **Re-train** | Re-treina a rede no browser |

---

### Gerar pesos pré-treinados

```bash
pip install numpy scipy matplotlib
python export_weights.py
```

Treina com `train-1000.csv` (7 epochs + augmentação ±10°) e gera:
- `weights.json` — para uso via `fetch` em webapps futuras
- `weights.js` — carregado diretamente pelo `index.html` sem CORS

---

### Treinamento interativo (Python)

```bash
python main.py
```

```
1 - Testar um dígito da folha de testes
2 - Medir desempenho geral
3 - Desenhar e reconhecer com draw_table
```

---

### Verificar pesos exportados

```bash
python verify.py   # requer weights.json e test-10.csv
node verify.js     # requer weights.json e test-10.csv
```

---

## ⚙️ Detalhes da Implementação

### Backpropagation

Os pesos **não** são transpostos em relação à formulação clássica do Tariq Rashid — a lógica é equivalente com outra convenção de orientação:

```
ΔWih = lr × (inputsᵀ · (hidden_errors × σ(H) × (1 − σ(H))))
ΔWho = lr × (hidden_outputsᵀ · (output_errors × σ(O) × (1 − σ(O))))
```

| Símbolo | Significado |
|:-------:|-------------|
| `lr` | Learning rate |
| `ᵀ` | Transposta |
| `σ` | Sigmoid |
| `H` | Saídas da camada oculta |
| `O` | Saídas da camada final |
| `×` | Multiplicação element-wise |
| `·` | Produto matricial |

---

## 📂 Estrutura de Arquivos

```
├── main.py              # Treinamento interativo e testes pelo terminal
├── export_weights.py    # Exporta pesos para uso na web
├── index.html           # Interface web (sem servidor necessário)
├── verify.py            # Verificação dos pesos em Python
├── verify.js            # Verificação dos pesos em Node.js
├── draw_table.pde       # Sketch Processing para coleta de dados
├── weights.json         # Pesos pré-treinados (gerado pelo export_weights.py)
├── weights.js           # Pesos em JS (carregado pelo index.html)
├── train-1000.csv       # 1.000 amostras de treino (MNIST)
├── train-100.csv        # 100 amostras de treino
├── test-10.csv          # 10 amostras de teste
└── number.csv           # Dígito desenhado via draw_table
```

---

## 📖 Referência

> Rashid, T. (2016). **Make Your Own Neural Network**. Independently published.
> ISBN: 978-1530826605

---

<div align="center">
<sub>Feito com ☕, numpy e muita cadeia de derivadas parciais</sub>
</div>
