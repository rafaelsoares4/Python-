# 📚 Índice Completo - Projeto MLP MNIST

Este documento serve como índice central para navegar por toda a documentação e scripts do projeto.

## 🎯 Início Rápido

**Nunca usou este projeto?** Comece aqui:

1. **[QUICKSTART.md](QUICKSTART.md)** ← Comece aqui! (5 minutos)
2. Execute `python quick_test.py` para validar instalação
3. Execute `python load_and_run_config.py --config baseline` para seu primeiro experimento

---

## 📖 Documentação

### Para Iniciantes
- **[README.md](README.md)** - Visão geral do projeto
- **[QUICKSTART.md](QUICKSTART.md)** - Guia de início rápido (5 min)
- **[requirements.txt](requirements.txt)** - Dependências necessárias

### Para Experimentação
- **[SIMULATION_SUGGESTIONS.md](SIMULATION_SUGGESTIONS.md)** - 10+ ideias de simulações prontas
- **[experiment_configs.json](experiment_configs.json)** - Configurações pré-definidas

### Para Usuários Avançados
- **[EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)** - Guia completo e detalhado
- **[INDEX.md](INDEX.md)** - Este arquivo (navegação geral)

---

## 🛠️ Scripts e Ferramentas

### Scripts de Execução

| Script | Quando Usar | Dificuldade | Tempo |
|--------|-------------|-------------|-------|
| **quick_test.py** | Validar instalação | ⭐ | 2-3 min |
| **load_and_run_config.py** | Usar configs prontas | ⭐ | 5-10 min |
| **run_custom_experiments.py** | Experimentos customizados | ⭐⭐ | 5-30 min |
| **advanced_experiments.py** | Suite completa | ⭐⭐⭐ | 30-45 min |
| **example_batch_simulations.py** | Exemplos programáticos | ⭐⭐ | 10-90 min |

### Scripts de Suporte

| Script | Propósito |
|--------|-----------|
| **main.py** | Experimentos básicos (originais) |
| **utils.py** | Funções auxiliares compartilhadas |

---

## 🎓 Caminhos de Aprendizado

### Caminho 1: Iniciante Absoluto
```
1. Leia: QUICKSTART.md
2. Execute: python quick_test.py
3. Execute: python load_and_run_config.py --config baseline
4. Analise: os gráficos em outputs_*/plots/
5. Próximo: SIMULATION_SUGGESTIONS.md (seção "Para Iniciantes")
```

### Caminho 2: Estudante de ML
```
1. Leia: README.md + QUICKSTART.md
2. Execute: python load_and_run_config.py --list
3. Execute: python load_and_run_config.py --configs baseline high_robustness
4. Leia: SIMULATION_SUGGESTIONS.md
5. Execute: python run_custom_experiments.py --experiments noise dropout
6. Analise: compare os resultados
7. Próximo: Crie seus próprios experimentos
```

### Caminho 3: Pesquisador/Avançado
```
1. Revise: EXPERIMENTS_GUIDE.md
2. Execute: python advanced_experiments.py
3. Analise: outputs_advanced/experiment_*/results/overall_summary.txt
4. Customize: Modifique advanced_experiments.py
5. Documente: Seus próprios insights
```

---

## 📊 Tipos de Experimentos Disponíveis

### Experimentos Sistemáticos

1. **Variação de Ruído** (`--experiments noise`)
   - 8 níveis de ruído (0.05 a 0.4)
   - Analisa degradação progressiva
   - Arquivo: `noise_levels_comparison.png`

2. **Variação de Learning Rate** (`--experiments learning_rate`)
   - 5 taxas diferentes (1e-4 a 1e-2)
   - Analisa convergência
   - Arquivo: `learning_rates_comparison.png`

3. **Variação de Batch Size** (`--experiments batch_size`)
   - 5 tamanhos (32 a 512)
   - Analisa trade-off tempo/qualidade
   - Arquivo: `batch_sizes_comparison.png`

4. **Variação de Dropout** (`--experiments dropout`)
   - 6 taxas (0.0 a 0.5)
   - Analisa regularização
   - Arquivo: `dropout_rates_comparison.png`

5. **Comparação de Ativações** (`--experiments activation`)
   - 4 funções (ReLU, Tanh, ELU, Sigmoid)
   - Analisa impact na convergência
   - Arquivo: `activations_comparison.png`

### Experimentos Customizados

Use `run_custom_experiments.py --custom` com parâmetros:
- `--initializer` (glorot, he, normal, constant)
- `--optimizer` (adam, sgd)
- `--learning-rate` (float)
- `--batch-size` (int)
- `--epochs` (int)
- `--dropout` (0.0 a 1.0)
- `--activation` (relu, tanh, elu, sigmoid)
- `--noise` (float)
- `--hidden-units` (int, padrão: 64)
- `--name` (string)

---

## 📁 Estrutura de Arquivos

### Arquivos Principais
```
projeto/
├── README.md                        # Visão geral
├── INDEX.md                         # Este arquivo
├── QUICKSTART.md                    # Início rápido
├── EXPERIMENTS_GUIDE.md             # Guia completo
├── SIMULATION_SUGGESTIONS.md        # Ideias de simulações
├── requirements.txt                 # Dependências
└── experiment_configs.json          # Configs pré-definidas
```

### Scripts Python
```
projeto/
├── quick_test.py                    # Teste de instalação
├── main.py                          # Experimentos básicos
├── advanced_experiments.py          # Suite completa
├── run_custom_experiments.py        # CLI para experimentos
├── load_and_run_config.py           # Executar configs JSON
├── example_batch_simulations.py     # Exemplos programáticos
└── utils.py                         # Funções auxiliares
```

### Notebooks (se existirem)
```
projeto/
├── mnist_robustness_analysis.ipynb  # Notebook principal
└── mnist_experiment.ipynb           # Experimentos adicionais
```

### Saídas
```
projeto/
├── outputs/                         # Experimentos básicos (main.py)
├── outputs_advanced/                # Experimentos avançados
├── outputs_quick_test/              # Testes rápidos
└── outputs_example_*/               # Exemplos específicos
```

---

## 🎯 Perguntas Frequentes

### Como começar?
```bash
python quick_test.py
```

### Como executar um experimento rápido?
```bash
python load_and_run_config.py --config baseline
```

### Como ver configurações disponíveis?
```bash
python load_and_run_config.py --list
```

### Como criar meu próprio experimento?
```bash
python run_custom_experiments.py --custom --dropout 0.3 --noise 0.2 --name "meu_teste"
```

### Como executar todos os experimentos?
```bash
python advanced_experiments.py
```

### Onde estão os resultados?
```
outputs_advanced/experiment_YYYYMMDD_HHMMSS/
├── plots/      # Gráficos PNG
├── results/    # JSON e TXT
└── models/     # Modelos salvos
```

### Como comparar múltiplas configurações?
```bash
python load_and_run_config.py --configs baseline high_robustness fast_training
```

### Como modificar parâmetros de um experimento sistemático?
Edite diretamente `advanced_experiments.py`, por exemplo:
```python
noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
```

---

## 🔍 Referência Rápida de Comandos

### Comandos Essenciais
```bash
# 1. Teste de instalação
python quick_test.py

# 2. Primeiro experimento
python load_and_run_config.py --config baseline

# 3. Ver opções disponíveis
python run_custom_experiments.py --help
python load_and_run_config.py --list

# 4. Experimento customizado simples
python run_custom_experiments.py --custom --dropout 0.2 --name "teste1"

# 5. Experimento sistemático
python run_custom_experiments.py --experiments noise

# 6. Suite completa
python advanced_experiments.py

# 7. Exemplos interativos
python example_batch_simulations.py
```

### Comandos por Objetivo

**Para entender robustez:**
```bash
python run_custom_experiments.py --experiments noise dropout
```

**Para otimizar treinamento:**
```bash
python run_custom_experiments.py --experiments learning_rate batch_size
```

**Para comparar arquiteturas:**
```bash
python run_custom_experiments.py --experiments activation
```

**Para análise completa:**
```bash
python advanced_experiments.py
```

---

## 📈 Métricas Coletadas

Cada experimento coleta:
- ✅ **clean_accuracy**: Accuracy em dados sem ruído
- ✅ **noisy_accuracy**: Accuracy em dados com ruído
- ✅ **degradation**: Perda percentual de accuracy
- ✅ **training_time**: Tempo de treinamento (segundos)
- ✅ **epochs_trained**: Número de épocas até convergência
- ✅ **final_val_loss**: Loss de validação final
- ✅ **final_val_accuracy**: Accuracy de validação final
- ✅ **history**: Histórico completo por época

---

## 🎨 Visualizações Geradas

### Gráficos Padrão
- Accuracy vs. Parâmetro (limpo e ruidoso)
- Degradação vs. Parâmetro
- Tempo de Treinamento vs. Parâmetro
- Tabela com métricas detalhadas

### Gráficos Especiais
- Heatmaps (noise × dropout)
- Curvas de convergência
- Comparações lado-a-lado
- Gráficos em escala logarítmica

---

## 🚀 Próximos Passos

Após dominar este projeto:

1. **Modifique a arquitetura**: Teste com 32, 128 neurônios
2. **Adicione camadas**: Teste MLP com 2 camadas ocultas
3. **Novos datasets**: CIFAR-10, Fashion-MNIST
4. **Novos tipos de ruído**: Salt & Pepper, Gaussian Blur
5. **Técnicas avançadas**: Batch Normalization, Layer Normalization
6. **Data Augmentation**: Rotações, translações
7. **Ensemble Methods**: Combine múltiplos modelos

---

## 📞 Suporte

### Problemas Comuns

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"Out of Memory"**
```bash
python run_custom_experiments.py --custom --batch-size 64
```

**Resultados diferentes do esperado**
- Verifique o seed (padrão: 42)
- Verifique os parâmetros de configuração
- Consulte os logs em `advanced_experiment.log`

### Recursos Adicionais
- Logs: `advanced_experiment.log`
- Configurações: `experiment_configs.json`
- Documentação TensorFlow: https://www.tensorflow.org/

---

## ✅ Checklist de Uso

- [ ] Li o QUICKSTART.md
- [ ] Executei quick_test.py com sucesso
- [ ] Executei meu primeiro experimento
- [ ] Entendi como ler os resultados
- [ ] Explorei diferentes configurações
- [ ] Executei experimentos sistemáticos
- [ ] Criei meus próprios experimentos
- [ ] Analisei e documentei os resultados

---

**🎉 Pronto para começar suas simulações!**

Para qualquer dúvida, consulte a documentação específica ou revise os exemplos em `example_batch_simulations.py`.

