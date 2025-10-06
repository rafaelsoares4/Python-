# 🔬 Análise de Robustez em Redes Neurais - MNIST

Este projeto demonstra o impacto de diferentes inicializações de pesos na robustez de redes neurais quando expostas a ruído gaussiano, utilizando o dataset MNIST.

## 📋 Descrição

O projeto compara quatro tipos de inicialização de pesos:
- **GlorotUniform (Xavier)**: Mantém variância constante entre camadas
- **HeUniform**: Otimizada para ativação ReLU
- **RandomNormal**: Distribuição normal simples
- **Constant**: Valor fixo para comparação

## 🚀 Funcionalidades

- ✅ Carregamento e pré-processamento do dataset MNIST
- ✅ Treinamento de modelos MLP com diferentes inicializações
- ✅ Adição de ruído gaussiano para teste de robustez
- ✅ Visualização comparativa de resultados
- ✅ Análise de degradação de performance por ruído

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
├── mnist_robustness_analysis.ipynb  # Notebook principal
├── main.py                          # Script de experimentos
├── utils.py                         # Funções auxiliares
├── requirements.txt                 # Dependências
├── README.md                        # Este arquivo
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

1. **Inicie o Jupyter Notebook:**
```bash
jupyter notebook mnist_robustness_analysis.ipynb
```

2. **Execute as células em ordem** para ver a análise completa

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

**Rafael Freitas**
- GitHub: [@rafaelsoares4](https://github.com/rafaelsoares4)

## 📚 Referências

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
