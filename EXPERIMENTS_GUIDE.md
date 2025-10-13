# 🧪 Guia de Experimentos Avançados - MLP MNIST (64 neurônios)

Este guia descreve os novos experimentos disponíveis para análise aprofundada da rede neural MLP com 64 neurônios na camada oculta.

## 📊 Experimentos Disponíveis

### 1. **Variação de Níveis de Ruído** 🔊
Testa a robustez da rede contra diferentes intensidades de ruído gaussiano.

- **Parâmetros testados**: σ = 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
- **Objetivo**: Identificar o ponto de ruptura de performance
- **Análise**: Curva de degradação vs. intensidade do ruído

### 2. **Variação de Taxas de Aprendizado** 📈
Investiga o impacto da taxa de aprendizado na convergência e robustez.

- **Parâmetros testados**: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
- **Objetivo**: Encontrar o equilíbrio entre velocidade de convergência e estabilidade
- **Análise**: Tempo de treinamento, accuracy final e overfitting

### 3. **Variação de Tamanhos de Batch** 📦
Analisa o efeito do batch size no treinamento e generalização.

- **Parâmetros testados**: 32, 64, 128, 256, 512
- **Objetivo**: Avaliar trade-off entre tempo de treinamento e qualidade do modelo
- **Análise**: Convergência, tempo de treinamento e estabilidade

### 4. **Variação de Taxas de Dropout** 🎯
Testa o efeito da regularização via dropout na robustez ao ruído.

- **Parâmetros testados**: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- **Objetivo**: Determinar se dropout melhora a robustez ao ruído gaussiano
- **Análise**: Trade-off entre overfitting e underfitting

### 5. **Variação de Funções de Ativação** ⚡
Compara diferentes funções de ativação na camada oculta.

- **Funções testadas**: ReLU, Tanh, ELU, Sigmoid
- **Objetivo**: Identificar qual ativação oferece melhor robustez
- **Análise**: Gradient flow, convergência e robustez

## 🚀 Como Executar

### Opção 1: Executar TODOS os Experimentos

```bash
python advanced_experiments.py
```

Isso executará todos os 5 experimentos sequencialmente e gerará:
- Resultados em JSON
- Relatórios em TXT
- Gráficos comparativos em PNG
- Resumo geral

**Tempo estimado**: ~30-45 minutos (depende do hardware)

### Opção 2: Executar Experimentos Específicos

```bash
# Executar apenas experimentos de ruído e dropout
python run_custom_experiments.py --experiments noise dropout

# Executar apenas experimento de learning rate
python run_custom_experiments.py --experiments learning_rate

# Executar todos
python run_custom_experiments.py --all
```

### Opção 3: Experimento Customizado

```bash
# Exemplo: Testar configuração específica
python run_custom_experiments.py --custom \
  --initializer he \
  --optimizer adam \
  --learning-rate 0.001 \
  --dropout 0.3 \
  --noise 0.25 \
  --activation relu \
  --batch-size 128 \
  --epochs 20 \
  --name "meu_experimento"
```

## 📂 Estrutura de Saídas

Cada execução cria uma pasta com timestamp:

```
outputs_advanced/
└── experiment_YYYYMMDD_HHMMSS/
    ├── models/              # Modelos salvos (.h5)
    ├── plots/               # Gráficos comparativos
    │   ├── noise_levels_comparison.png
    │   ├── learning_rates_comparison.png
    │   ├── batch_sizes_comparison.png
    │   ├── dropout_rates_comparison.png
    │   └── activations_comparison.png
    ├── results/             # Resultados em JSON e TXT
    │   ├── noise_levels_results.json
    │   ├── noise_levels_summary.txt
    │   ├── learning_rates_results.json
    │   ├── learning_rates_summary.txt
    │   ├── batch_sizes_results.json
    │   ├── batch_sizes_summary.txt
    │   ├── dropout_rates_results.json
    │   ├── dropout_rates_summary.txt
    │   ├── activations_results.json
    │   ├── activations_summary.txt
    │   └── overall_summary.txt
    └── logs/                # Logs de execução
```

## 📈 Métricas Analisadas

Para cada experimento, as seguintes métricas são coletadas:

1. **Accuracy em dados limpos** (baseline)
2. **Accuracy em dados ruidosos** (robustez)
3. **Degradação percentual** (clean_acc - noisy_acc)
4. **Tempo de treinamento** (segundos)
5. **Número de épocas** (até early stopping)
6. **Loss de validação final**
7. **Histórico completo** (loss e accuracy por época)

## 🎨 Visualizações Geradas

Cada experimento gera um gráfico com 4 painéis:

1. **Top-left**: Accuracy vs. Parâmetro (limpo e ruidoso)
2. **Top-right**: Degradação vs. Parâmetro
3. **Bottom-left**: Tempo de Treinamento vs. Parâmetro
4. **Bottom-right**: Tabela com métricas detalhadas

## 💡 Dicas de Uso

### Para Análise Rápida
```bash
# Execute apenas o experimento de interesse
python run_custom_experiments.py --experiments noise
```

### Para Análise Completa
```bash
# Execute todos os experimentos
python advanced_experiments.py
```

### Para Teste de Hipótese Específica
```bash
# Teste uma configuração específica
python run_custom_experiments.py --custom \
  --dropout 0.4 \
  --noise 0.3 \
  --name "teste_dropout_alto"
```

## 📊 Exemplos de Análises

### Exemplo 1: Encontrar o Melhor Learning Rate
```bash
python run_custom_experiments.py --experiments learning_rate
# Analise os gráficos em outputs_advanced/experiment_*/plots/learning_rates_comparison.png
```

### Exemplo 2: Testar Regularização
```bash
python run_custom_experiments.py --experiments dropout
# Compare degradação com e sem dropout
```

### Exemplo 3: Robustez Extrema
```bash
# Teste com ruído muito alto
python run_custom_experiments.py --custom \
  --noise 0.5 \
  --dropout 0.3 \
  --name "robustez_extrema"
```

## 🔍 Interpretação dos Resultados

### Degradação Baixa (< 5%)
- Modelo muito robusto ao ruído
- Boa generalização
- Provavelmente não está overfitting

### Degradação Média (5-15%)
- Robustez aceitável
- Comportamento esperado para MNIST
- Trade-off razoável

### Degradação Alta (> 15%)
- Modelo sensível ao ruído
- Pode estar overfitting
- Considere regularização adicional

## 🧠 Configurações Recomendadas

Com base nos experimentos, configurações típicas bem-sucedidas:

### Para Máxima Accuracy
```python
initializer='he'
optimizer='adam'
learning_rate=1e-3
batch_size=128
dropout=0.0
activation='relu'
```

### Para Máxima Robustez
```python
initializer='he'
optimizer='adam'
learning_rate=5e-4
batch_size=64
dropout=0.2
activation='relu'
```

### Para Treinamento Rápido
```python
initializer='he'
optimizer='adam'
learning_rate=1e-3
batch_size=256
dropout=0.1
activation='relu'
```

## 📚 Referências e Insights

### Sobre Inicialização
- **He**: Melhor para ReLU e ELU
- **Glorot**: Melhor para Tanh e Sigmoid

### Sobre Dropout
- Dropout alto (>0.3) pode prejudicar em redes pequenas
- Dropout moderado (0.2-0.3) geralmente melhora robustez

### Sobre Learning Rate
- LR muito alto: convergência instável
- LR muito baixo: treinamento lento
- Sweet spot geralmente entre 5e-4 e 1e-3

### Sobre Batch Size
- Batch pequeno: mais ruído no gradiente, pode generalizar melhor
- Batch grande: convergência mais suave, mais rápido por época

## ⚙️ Configurações Avançadas

Para modificar experimentos, edite diretamente o arquivo `advanced_experiments.py`:

```python
# Exemplo: Adicionar mais níveis de ruído
noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Exemplo: Testar L2 regularization
config['regularizer'] = 'l2'
config['reg_strength'] = 0.01
```

## 🐛 Troubleshooting

### Erro de Memória
- Reduza o batch size
- Reduza o número de experimentos simultâneos

### Treinamento Muito Lento
- Aumente o batch size
- Reduza validation_split
- Use GPU se disponível

### Early Stopping Muito Cedo
- Aumente patience
- Reduza min_delta
- Ajuste learning rate

## 📞 Suporte

Para dúvidas ou sugestões:
- Abra uma issue no GitHub
- Consulte a documentação do TensorFlow
- Revise os logs em `advanced_experiment.log`

---

**Desenvolvido para análise sistemática de redes neurais MLP** 🚀

