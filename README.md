# 🔬 Análise de Robustez em Redes Neurais - MNIST

Este projeto demonstra o impacto de diferentes inicializações de pesos na robustez de redes neurais quando expostas a ruído gaussiano, utilizando o dataset MNIST.

## 📋 Descrição

O projeto compara quatro tipos de inicialização de pesos:
- **GlorotUniform (Xavier)**: Mantém variância constante entre camadas
- **HeUniform**: Otimizada para ativação ReLU
- **RandomNormal**: Distribuição normal simples
- **Constant**: Valor fixo para comparação

## 🚀 Funcionalidades

### Experimentos Básicos
- ✅ Carregamento e pré-processamento do dataset MNIST
- ✅ Treinamento de modelos MLP com diferentes inicializações
- ✅ Adição de ruído gaussiano para teste de robustez
- ✅ Visualização comparativa de resultados
- ✅ Análise de degradação de performance por ruído

### 🆕 Experimentos Avançados (Novo!)
- ✅ **Variação de Níveis de Ruído**: Testa 8 níveis diferentes (0.05 a 0.4)
- ✅ **Variação de Learning Rates**: 5 taxas diferentes (1e-4 a 1e-2)
- ✅ **Variação de Batch Sizes**: 5 tamanhos (32 a 512)
- ✅ **Testes com Dropout**: 6 taxas de dropout (0.0 a 0.5)
- ✅ **Funções de Ativação**: Compara ReLU, Tanh, ELU, Sigmoid
- ✅ **Análise Sistemática**: 30+ experimentos automatizados
- ✅ **Visualizações Avançadas**: Gráficos multi-painel com análise detalhada

## 📊 Resultados

O notebook gera gráficos comparativos mostrando:
- Accuracy em dados limpos vs. ruidosos
- Degradação percentual de performance
- Tabela detalhada com métricas

## 🛠️ Tecnologias

- **Python 3.11**
- **TensorFlow 2.16.2**
- **NumPy**
- **Matplotlib**
- **Jupyter Notebook**

## 📁 Estrutura do Projeto

```
├── mnist_robustness_analysis.ipynb  # Notebook principal (experimentos básicos)
├── main.py                          # Script de experimentos básicos
├── advanced_experiments.py          # 🆕 Script de experimentos avançados
├── run_custom_experiments.py        # 🆕 Interface CLI para experimentos
├── utils.py                         # Funções auxiliares
├── requirements.txt                 # Dependências
├── README.md                        # Este arquivo
├── EXPERIMENTS_GUIDE.md             # 🆕 Guia detalhado de experimentos
└── .gitignore                       # Arquivos ignorados pelo Git
```

## 🔧 Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/rafaelsoares4/neural-network-robustness-analysis.git
cd neural-network-robustness-analysis
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Como Executar

### Experimentos Básicos

1. **Via Jupyter Notebook:**
```bash
jupyter notebook mnist_robustness_analysis.ipynb
```

2. **Via Script Python:**
```bash
python main.py
```

### 🆕 Experimentos Avançados

1. **Executar TODOS os experimentos avançados:**
```bash
python advanced_experiments.py
```
*Tempo estimado: 30-45 minutos*

2. **Executar experimentos específicos:**
```bash
# Apenas ruído e dropout
python run_custom_experiments.py --experiments noise dropout

# Apenas learning rate
python run_custom_experiments.py --experiments learning_rate

# Todos os experimentos
python run_custom_experiments.py --all
```

3. **Experimento customizado:**
```bash
python run_custom_experiments.py --custom \
  --initializer he \
  --optimizer adam \
  --learning-rate 0.001 \
  --dropout 0.3 \
  --noise 0.25 \
  --batch-size 128 \
  --epochs 20 \
  --name "meu_teste"
```

4. **Ver todas as opções:**
```bash
python run_custom_experiments.py --help
```

Para mais detalhes, consulte: **[EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)**

## 📈 Resultados Esperados

- **HeUniform**: Geralmente melhor performance com ReLU
- **GlorotUniform**: Boa performance geral
- **RandomNormal**: Performance variável
- **Constant**: Baseline (geralmente pior performance)

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novas funcionalidades
- Melhorar a documentação

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

**Rafael Soares**
- GitHub: [@rafaelsoares4](https://github.com/rafaelsoares4)

## 📖 Documentação Completa

Este projeto inclui documentação extensa para diferentes níveis de usuário:

### 🚀 Para Começar Rapidamente
- **[QUICKSTART.md](QUICKSTART.md)** - Guia de início rápido (5 minutos)
  - Teste de instalação
  - Primeiros experimentos
  - Comandos essenciais

### 🧪 Para Experimentar
- **[SIMULATION_SUGGESTIONS.md](SIMULATION_SUGGESTIONS.md)** - Sugestões específicas de simulações
  - 10 objetivos de pesquisa diferentes
  - Simulações prontas para copiar/colar
  - Análise por cenário

### 📊 Para Entender em Profundidade
- **[EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)** - Guia completo de experimentos
  - Detalhes de cada tipo de experimento
  - Interpretação de resultados
  - Configurações avançadas

### 🎯 Fluxo Recomendado

```mermaid
Iniciante → QUICKSTART.md → Execute quick_test.py
    ↓
Intermediário → SIMULATION_SUGGESTIONS.md → Execute configurações pré-definidas
    ↓
Avançado → EXPERIMENTS_GUIDE.md → Crie experimentos customizados
```

## 🛠️ Scripts Disponíveis

| Script | Propósito | Tempo | Dificuldade |
|--------|-----------|-------|-------------|
| `quick_test.py` | Validar instalação | 2-3 min | ⭐ Fácil |
| `load_and_run_config.py` | Usar configs pré-definidas | 5-10 min | ⭐ Fácil |
| `run_custom_experiments.py` | Experimentos customizados | 5-30 min | ⭐⭐ Médio |
| `advanced_experiments.py` | Suite completa | 30-45 min | ⭐⭐⭐ Avançado |

## 📚 Referências

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
