# ⚡ Início Rápido - Experimentos MLP MNIST (64 neurônios)

Este guia rápido mostra como começar a executar simulações imediatamente.

## 📋 Pré-requisitos

```bash
# Certifique-se de ter Python 3.11+ instalado
python --version

# Instale as dependências
pip install -r requirements.txt
```

## 🧪 Teste Rápido (2-3 minutos)

Antes de executar experimentos completos, faça um teste rápido:

```bash
python quick_test.py
```

Este script irá:
- ✅ Verificar se todas as dependências estão instaladas
- ✅ Executar um treinamento rápido (3 épocas)
- ✅ Validar que tudo está funcionando

## 🚀 Executar Simulações

### Opção 1: Usar Configurações Pré-definidas (RECOMENDADO)

**Passo 1**: Ver configurações disponíveis
```bash
python load_and_run_config.py --list
```

**Passo 2**: Executar uma configuração
```bash
# Configuração baseline
python load_and_run_config.py --config baseline

# Alta robustez
python load_and_run_config.py --config high_robustness

# Treinamento rápido
python load_and_run_config.py --config fast_training
```

**Passo 3**: Comparar múltiplas configurações
```bash
python load_and_run_config.py --configs baseline high_robustness fast_training
```

### Opção 2: Experimentos Sistemáticos

**Teste um parâmetro específico:**

```bash
# Apenas níveis de ruído
python run_custom_experiments.py --experiments noise

# Apenas dropout
python run_custom_experiments.py --experiments dropout

# Múltiplos parâmetros
python run_custom_experiments.py --experiments noise dropout learning_rate
```

**Execute TODOS os experimentos sistemáticos:**

```bash
python advanced_experiments.py
```
⚠️ **Atenção**: Isso pode levar 30-45 minutos!

### Opção 3: Experimento Customizado

Crie sua própria configuração:

```bash
python run_custom_experiments.py --custom \
  --initializer he \
  --optimizer adam \
  --learning-rate 0.001 \
  --dropout 0.2 \
  --noise 0.3 \
  --batch-size 128 \
  --epochs 15 \
  --activation relu \
  --name "meu_experimento"
```

## 📊 Sugestões de Simulações para Começar

### 1️⃣ Comparação de Robustez Básica
```bash
python load_and_run_config.py --configs baseline high_robustness extreme_noise
```
**O que isso testa**: Diferentes níveis de robustez ao ruído

### 2️⃣ Comparação de Funções de Ativação
```bash
python load_and_run_config.py --configs baseline tanh_activation elu_activation
```
**O que isso testa**: Impacto da função de ativação

### 3️⃣ Impacto do Learning Rate
```bash
python load_and_run_config.py --configs baseline low_learning_rate high_learning_rate
```
**O que isso testa**: Como a taxa de aprendizado afeta o treinamento

### 4️⃣ Impacto do Batch Size
```bash
python load_and_run_config.py --configs baseline small_batch large_batch
```
**O que isso testa**: Trade-off entre tempo e qualidade

### 5️⃣ Regularização
```bash
python load_and_run_config.py --configs baseline high_regularization
```
**O que isso testa**: Efeito de dropout alto

## 📁 Onde Encontrar os Resultados

Após executar um experimento, os resultados estarão em:

```
outputs_advanced/experiment_YYYYMMDD_HHMMSS/
├── plots/              # 📊 Gráficos PNG
├── results/            # 📄 JSON e TXT com métricas
└── models/             # 🤖 Modelos salvos (se habilitado)
```

**Arquivos importantes:**
- `plots/*_comparison.png` - Gráficos comparativos
- `results/*_summary.txt` - Resumo em texto
- `results/*_results.json` - Dados detalhados

## 🎯 Fluxo de Trabalho Recomendado

### Para Iniciantes

1. **Teste rápido**
   ```bash
   python quick_test.py
   ```

2. **Execute configuração baseline**
   ```bash
   python load_and_run_config.py --config baseline
   ```

3. **Compare com configuração de alta robustez**
   ```bash
   python load_and_run_config.py --config high_robustness
   ```

4. **Analise os resultados** nos arquivos PNG e TXT

### Para Usuários Intermediários

1. **Execute experimentos sistemáticos específicos**
   ```bash
   python run_custom_experiments.py --experiments noise dropout
   ```

2. **Analise os gráficos gerados**

3. **Crie configurações customizadas** baseadas nos insights

### Para Usuários Avançados

1. **Execute suite completa**
   ```bash
   python advanced_experiments.py
   ```

2. **Crie arquivo de configuração customizado** (cópia de `experiment_configs.json`)

3. **Execute análises comparativas** com seus próprios parâmetros

## 💡 Dicas Importantes

### ⚡ Performance
- Use **batch sizes maiores** (256-512) para treinar mais rápido
- Reduza **epochs** para testes iniciais
- Use **GPU** se disponível (TensorFlow detecta automaticamente)

### 🎨 Visualização
- Todos os gráficos são salvos em alta resolução (300 DPI)
- Abra os arquivos `.png` em `plots/` para análise visual
- Use os arquivos `.json` para análises programáticas

### 📝 Logs
- Logs detalhados em `advanced_experiment.log`
- Use para debug se algo der errado

### 💾 Armazenamento
- Cada experimento completo gera ~10-50 MB
- Modelos salvos ocupam ~1-5 MB cada
- Desabilite `save_model` se quiser economizar espaço

## 🆘 Problemas Comuns

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Out of Memory"
```bash
# Reduza o batch size
python run_custom_experiments.py --custom --batch-size 64
```

### Treinamento muito lento
```bash
# Use batch maior ou menos épocas
python run_custom_experiments.py --custom --batch-size 256 --epochs 10
```

### Resultados ruins
- Verifique se o learning rate está adequado
- Tente diferentes inicializadores
- Aumente o número de épocas

## 📚 Próximos Passos

Após dominar os experimentos básicos:

1. Leia **[EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)** para detalhes aprofundados
2. Explore **[README.md](README.md)** para contexto do projeto
3. Modifique `advanced_experiments.py` para criar seus próprios experimentos

## 🎓 Experimentos Sugeridos para Estudo

### Para entender Robustez
```bash
python run_custom_experiments.py --experiments noise
```

### Para entender Overfitting
```bash
python run_custom_experiments.py --experiments dropout
```

### Para entender Otimização
```bash
python run_custom_experiments.py --experiments learning_rate batch_size
```

### Para entender Arquitetura
```bash
python run_custom_experiments.py --experiments activation
```

---

## 🎉 Comece Agora!

**Comando mais simples para começar:**

```bash
python quick_test.py && python load_and_run_config.py --config baseline
```

Isso fará:
1. ✅ Teste de instalação (2 minutos)
2. ✅ Primeiro experimento completo (5-10 minutos)
3. ✅ Resultados prontos para análise

**Boa sorte com suas simulações! 🚀**

