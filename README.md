# Neural Network — Reconhecimento de Dígitos

Rede neural implementada **do zero em Python** para reconhecer dígitos manuscritos (0–9), treinada com o dataset MNIST.
Baseada no livro **"Make Your Own Neural Network"** de _Tariq Rashid_, com adaptações próprias na organização dos pesos e na adição de data augmentation.

---

## Visão geral

O projeto é dividido em duas partes independentes:

| Arquivo | Tecnologia | Responsabilidade |
|---|---|---|
| `main.py` | Python 3 | Toda a lógica da rede neural: definição, treinamento, backpropagation e inferência |
| `draw_table.pde` | Processing (Java) | **Somente interface gráfica** — janela de desenho para o usuário escrever um dígito |

> **A inteligência fica inteiramente no Python.**  
> O Processing é utilizado apenas para capturar o desenho do usuário em uma grade 28×28, salvar os valores de pixel no arquivo `number.csv` e encerrar. Após isso, o Python lê o arquivo e classifica o dígito.

---

## Como a rede funciona

### Arquitetura

```
Entrada (784)  →  Camada oculta (200)  →  Saída (10)
   28×28 px         sigmoid                 0 – 9
```

- **Pesos** inicializados com distribuição normal: média 0, desvio padrão `1/√n` (onde `n` é o número de entradas da camada).
- **Função de ativação:** Sigmoid (`scipy.special.expit`) em todas as camadas.
- **Saída:** o índice do neurônio com maior valor é o dígito reconhecido.

### Treinamento (forward + backpropagation)

1. **Forward pass:** os inputs percorrem `Entrada → Oculta → Saída` via produto matricial e sigmoid.
2. **Cálculo de erros:**
   - Erro na saída: `E_out = target − output`
   - Erro na camada oculta: `E_hidden = E_out · Whoᵀ`
3. **Atualização de pesos** (gradiente descendente):
   - `ΔW = lr × (camada_anterior)ᵀ · (erro × sigmoid(saída) × (1 − sigmoid(saída)))`
   - Nota: diferentemente do livro do Tariq, os pesos **não são transpostos** na definição das matrizes — a lógica é equivalente, mas com outra convenção de orientação.

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Neurônios de entrada | 784 |
| Neurônios ocultos | 200 |
| Neurônios de saída | 10 |
| Taxa de aprendizado | 0.01 |
| Épocas | 5 |

### Data Augmentation

Para cada amostra de treino, a rede é treinada três vezes:
1. Imagem original
2. Imagem rotacionada **+10°**
3. Imagem rotacionada **−10°**

Isso aumenta artificialmente o conjunto de treino e melhora a generalização.

---

## Dataset

Os dados seguem o formato CSV do MNIST: cada linha começa com o **rótulo** (0–9) seguido de **784 valores** de pixel (0–255).

| Arquivo | Conteúdo |
|---|---|
| `train-1000.csv` | 1000 amostras para treinamento |
| `train-100.csv` | 100 amostras (subconjunto menor) |
| `test-10.csv` | 10 amostras para teste rápido |
| `number.csv` | Gerado pelo Processing com o desenho do usuário |

---

## Interface gráfica (Processing)

O arquivo `draw_table.pde` é um sketch em **Processing (Java)** que:

1. Abre uma janela 280×280 pixels dividida em grade 28×28.
2. Permite ao usuário **desenhar com o mouse** (com efeito de suavização de borda).
3. Possui dois botões:
   - **Done** (verde, canto esquerdo): salva o desenho em `number.csv` e fecha a janela.
   - **Clean** (vermelho, canto direito): limpa a tela para recomeçar.
4. Ao fechar, cada célula da grade tem seu valor de opacidade (0–255) escrito como CSV de 784 valores.

O Python lança o executável do Processing via `subprocess.run` e, logo após o fechamento da janela, lê o `number.csv` gerado para classificar o dígito.

---

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

---

## Como usar

**1. Execute o script principal:**

```bash
python main.py
```

O treinamento inicia automaticamente (5 épocas sobre `train-1000.csv`, ~15 000 chamadas de treino com augmentation).

**2. Escolha uma opção no menu interativo:**

```
1 - Um dígito da folha de testes
2 - Teste de desempenho
3 - Desenhe você mesmo
0 - Sair
```

- **Opção 1:** informa um índice (0–9) e exibe o dígito correspondente do arquivo de teste junto com a predição da rede.
- **Opção 2:** percorre todo o `train-1000.csv` e exibe a porcentagem de acerto (performance score).
- **Opção 3:** abre a janela do Processing. Desenhe o dígito, clique em **Done** e o Python exibirá o número reconhecido.

---

## Referência

> Rashid, T. (2016). **Make Your Own Neural Network**. Independently published.  
> ISBN: 978-1530826605
