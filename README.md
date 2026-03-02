<div align="center">

# 🧠 Neural Network — Reconhecimento de Dígitos

*Rede neural do zero, sem frameworks, capaz de ler sua caligrafia*

[![Language](https://img.shields.io/badge/Language-Python%203-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Interface](https://img.shields.io/badge/Interface-Processing-FF6B6B?style=flat-square)](https://processing.org)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-00d4aa?style=flat-square)](http://yann.lecun.com/exdb/mnist/)
[![Algorithm](https://img.shields.io/badge/Algoritmo-Backpropagation-00d4aa?style=flat-square)]()

</div>

---

Rede neural implementada **do zero em Python** para reconhecer dígitos manuscritos (0–9), treinada com o dataset MNIST.
Baseada no livro **"Make Your Own Neural Network"** de *Tariq Rashid*, com adaptações próprias na organização dos pesos e na adição de data augmentation.

---

## 📐 Visão Geral

| Arquivo | Tecnologia | Responsabilidade |
|---------|-----------|-----------------|
| `main.py` | Python 3 | Toda a lógica da rede neural: definição, treinamento, backpropagation e inferência |
| `draw_table.pde` | Processing (Java) | **Somente interface gráfica** — janela de desenho para o usuário escrever um dígito |

> **A inteligência fica inteiramente no Python.**
> O Processing captura o desenho em uma grade 28×28, salva os pixels em `number.csv` e encerra. O Python lê o arquivo e classifica o dígito.

---

## 🧬 Como a Rede Funciona

### Arquitetura

```
Entrada (784)  →  Camada Oculta (200)  →  Saída (10)
   28×28 px           sigmoid               0 – 9
```

- **Pesos** inicializados com distribuição normal: média `0`, desvio padrão `1/√n`
- **Função de ativação:** Sigmoid em todas as camadas
- **Saída:** índice do neurônio com maior valor = dígito reconhecido

---

### Treinamento (Forward + Backpropagation)

**1. Forward pass**

Os inputs percorrem `Entrada → Oculta → Saída` via produto matricial e sigmoid.

**2. Cálculo de erros**

- Erro na saída: `E_out = target − output`
- Erro na camada oculta: `E_hidden = E_out · Whoᵀ`

**3. Atualização de pesos** (gradiente descendente)

```
ΔW = lr × (camada_anterior)ᵀ · (erro × sigmoid(saída) × (1 − sigmoid(saída)))
```

> **Nota:** diferentemente do livro do Tariq, os pesos **não são transpostos** na definição das matrizes — a lógica é equivalente, mas com outra convenção de orientação.

---

### Hiperparâmetros

| Parâmetro | Valor |
|-----------|:-----:|
| Neurônios de entrada | `784` |
| Neurônios ocultos | `200` |
| Neurônios de saída | `10` |
| Taxa de aprendizado | `0.01` |
| Épocas | `5` |

---

### Data Augmentation

Para cada amostra de treino, a rede é treinada **três vezes**:

1. Imagem original
2. Imagem rotacionada **+10°**
3. Imagem rotacionada **−10°**

Isso triplica artificialmente o conjunto de treino e melhora a generalização.

---

## 📊 Dataset

Formato CSV do MNIST: cada linha começa com o **rótulo** (0–9) seguido de **784 valores** de pixel (0–255).

| Arquivo | Conteúdo |
|---------|----------|
| `train-1000.csv` | 1000 amostras para treinamento |
| `train-100.csv` | 100 amostras (subconjunto menor) |
| `test-10.csv` | 10 amostras para teste rápido |
| `number.csv` | Gerado pelo Processing com o desenho do usuário |

---

## 🖼️ Interface Gráfica (Processing)

O arquivo `draw_table.pde` abre uma janela **280×280 px** dividida em grade 28×28 onde o utilizador:

1. **Desenha com o mouse** (com efeito de suavização de borda)
2. Clica em **Done** (verde) → salva `number.csv` e fecha
3. Clica em **Clean** (vermelho) → limpa a tela para recomeçar

O Python lança o Processing via `subprocess.run` e, após fechar a janela, lê o CSV gerado para classificar o dígito.

---

## 📦 Dependências

```bash
pip install numpy scipy matplotlib
```

---

## 🚀 Como Usar

**1. Execute o script principal:**

```bash
python main.py
```

O treinamento inicia automaticamente — 5 épocas sobre `train-1000.csv`, ~15.000 chamadas com augmentation.

**2. Escolha uma opção no menu:**

```
1 - Um dígito da folha de testes
2 - Teste de desempenho
3 - Desenhe você mesmo
0 - Sair
```

| Opção | Descrição |
|:-----:|-----------|
| `1` | Informa um índice (0–9) e exibe o dígito do arquivo de teste com a predição da rede |
| `2` | Percorre todo o `train-1000.csv` e exibe a porcentagem de acerto |
| `3` | Abre a janela do Processing — desenhe, clique em **Done** e veja o resultado |

---

## 📖 Referência

> Rashid, T. (2016). **Make Your Own Neural Network**. Independently published.
> ISBN: 978-1530826605

---

<div align="center">
<sub>Feito com ☕, numpy e muita cadeia de derivadas parciais</sub>
</div>
