<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript Badge"/>
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5 Badge"/>
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3 Badge"/>
</div>

<h1 align="center">🧠 Neural Network Digit Classifier</h1>

<p align="center">
  <strong>Uma Rede Neural Artificial construída do absoluto zero (sem frameworks como PyTorch ou TensorFlow) capaz de reconhecer dígitos desenhados à mão no navegador com alta precisão.</strong>
</p>

---

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Arquitetura da Rede](#-arquitetura-da-rede)
- [Funcionalidades e Otimizações Inovadoras](#-funcionalidades-e-otimizações-inovadoras)
- [Como Executar o Projeto](#-como-executar-o-projeto)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Entendendo os Pesos e Treinamento](#-entendendo-os-pesos-e-treinamento)

---

## 👁️ Visão Geral

Este projeto demonstra os fundamentos de **Machine Learning** e **Deep Learning** escrevendo uma Rede Neural Perceptron Multicamadas (MLP) da estaca zero através das linguagens Python (para o treinamento back-end) e JavaScript (para a inferência no front-end em tempo real).

Ele aborda um problema clássico: o reconhecimento de caracteres manuscritos, utilizando uma das mais famosas bases de dados matemáticas — o dataset **MNIST**.

### Por que "do zero"?
Ao longo do código de `main.py` e dentro do próprio `index.html` via JS, você encontrará os cálculos puros algébricos (multiplicação de matrizes, derivadas, descida de gradiente) documentados ao invés de meramente chamar métodos de bibliotecas abstratas. É um projeto focado no ensino profundo de como as IAs realmente pensam e tomam decisões.

---

## � Arquitetura da Rede

Nossa rede neural estática clássica feed-forward é definida estruturalmente como:
**784 → 200 → 10**

- **Camada de Entrada (Input Layer): 784 Nodes**
  - O quadro principal de desenho HTML `<canvas>` possui originalmente dimensões lógicas transformadas na matriz 28x28. Cada pixel em tons de cinza vira um input contínuo (de 0.01 a 0.99).
- **Camada Oculta (Hidden Layer): 200 Nodes**
  - A camada intermediária extrai padrões primitivos (linhas, curvas, arestas) utilizando funções de ativação contínuas.
- **Camada de Saída (Output Layer): 10 Nodes**
  - Cada *Node* (Nó) final gera a probabilidade (Confiança de X%) do desenho representar os numerais de **0 a 9**.
- **Função de Ativação:** Utilizada a **Sigmóide** em ambas as camadas de aprendizado (`scp.expit` no Python, `1 / (1 + Math.exp(-x))` no JS).
- **Cálculo de Erros e Treino:** Baseado em Regra Delta, onde o Backpropagation manual corrige o erro da Camada *Output* via pesos W<sub>ho</sub> transpostos para calcular o erro da Camada Oculta.

---

## � Funcionalidades e Otimizações Inovadoras

- **Center of Mass Normalization (Processamento MNIST em Tempo Real):**
  - Redes estáticas (sem camadas convolucionais iterativas) dependem da geometria estrita do desenho. O maior vilão da acurácia front-end costuma ser a inconstância da posição humana.
  - Implementamos um **Algoritmo de Centro de Massa Bounding-Box** em JavaScript puro (`preProcessImage`). A cada classificação, a rede traça uma "caixa" (bounding_box) visível englobando todo o escopo do usuário, reduz essa caixa sob interpolação bilinear perfeitamente para 20x20px e calcula a massa para reposicionar todo o conjunto exatamente nas coordenadas centrais do quadro final (28x28). Isso reflete perfeitamente como os dados do Dataset MNIST treinaram a IA!
- **Pesos Globais Injetados sem Web Servers:**
  - Carregar arquivos locais `.json` dentro do navegador dispara alertas clássicos de bloqueio CORS em execuções do protocolo `file:///`.
  - Criamos uma engenharia na rotina do programa (`export_weights.py`) capaz de salvar as grandes matrizes sintáticas diretamente em Javascript assíncrono interpretável (`weights.js`), viabilizando com que qualquer leigo abra o projeto localmente com dois cliques no HTML sem falhas e sem rodar localhosts.
- **Data Augmentation:**
  - O loop de treinamento (`main.py`) faz uso orgânico de transformações físicas em dados. Rotacionalmente distorce matrizes matriz em ângulos variados (+10°, -10°) via scypi provendo um exército de pesos muito mais robusto e não-triviais ao modelo base.

---

## 🕹️ Como Executar o Projeto

**1. Usando Pelo Navegador (Uso Padrão - Frente de Inferência):**
O site foi configurado para ser plug-and-play completamente autossuficiente caso os pesos pré-treinados já tenham sido gerados e exportados para `weights.js`.
* Descompacte os arquivos num diretório normal.
* Dê um duplo clique no seu arquivo `index.html` pelo seu explorador de arquivos e use-o livremente!
* Destaques de UX (User Experience): O sistema tem componentes responsivos na tela que mudam as barras de confiança dos números usando degradês e possui feedback estrito de acurácia colorida. Você também ganha o Live Preview de como a IA trata o "Centro da Massa" da sua tinta no cantinho da tela 28x28 input.

---

## 📚 Entendendo os Pesos e Treinamento

Se desejar mergulhar fundo e retreinar manualmente o cérebro da rede neural (ou aplicar Hyperparameter tuning):

### Pré-requisitos
Certifique-se que o ambiente tem as compilações Python base matemáticas:
```bash
pip install numpy scipy matplotlib
```

### Script Interativo de Aprendizado (`main.py`):
Este é o core principal usado para treinar a rede e testar sua capacidade individual pelo terminal contra testes aleatórios do Dataset original ou desenho na tela via executável paralelo `draw_table`. Ele não exportará nada final. Retorna um relatório de porcentagens de "Performance".
```bash
python main.py
```

### Exportação Comercial / Deployment (`export_weights.py`):
Quando você encontrar a taxa de aprendizagem (Learning Factor) ou quantia de Nodes ideais, utilize o exportador estático. 
Ele varre o `train-1000.csv` inteiramente, aplica _Data Augmentation_ e gera dois arquivos vitais:
* `weights.json`: Base crua estática utilizável caso vá interagir via `fetch` de API na sua WebApp futura.
* `weights.js`: A base consumível em tempo interativo e dinâmico, exportando uma variável nativa lida de forma autônoma e imediata pelo `index.html` em modo local!

```bash
python export_weights.py
```
> _O Script é um processo intenso computacional que relata, rodada a rodada, as predições completadas por amostragem base._

---

## 📂 Estrutura do Projeto

```text
Neural-Network-1/
├── index.html                 # Aplicativo Web Front-end Moderno interagível
├── main.py                    # Back-end script interativo de Treinamento Geral
├── export_weights.py          # Back-end Script para Gerar Pesos exportáveis (JSON e JS)
├── train-*.csv                # Base de dados (MNIST reduzida) de treinamento e label
├── test-10.csv                # Pequena base para testing cego iterável do core
├── weights.json               # (Gerado) Estado cerebral Neural serializado
└── weights.js                 # (Gerado) Estado cerebral exposto para contorno de CORS
```

---

<p align="center">
Desenvolvido com curiosidade, em Javascript e Python!
</p>
