# main.py
# Script principal para treinar e comparar inicializações no MLP com MNIST

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.initializers import RandomNormal, Constant, GlorotUniform, HeUniform
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import preprocess_mnist, add_gaussian_noise, plot_history, evaluate_model_performance

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configurações do experimento."""
    batch_size: int = 128
    epochs: int = 15
    hidden_units: int = 64
    learning_rate: float = 1e-3
    noise_std: float = 0.2
    seed: int = 42
    validation_split: float = 0.1
    patience: int = 3
    min_delta: float = 0.001

    def to_dict(self):
        return asdict(self)


# Configurações globais
CONFIG = ExperimentConfig()

# Configurar seeds
np.random.seed(CONFIG.seed)
tf.random.set_seed(CONFIG.seed)


def setup_experiment_directories(base_dir: str = "outputs") -> Dict[str, Path]:
    """Cria diretórios para organizar os resultados do experimento."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(base_dir) / f"experiment_{timestamp}"

    directories = {
        'base': base_path,
        'models': base_path / "models",
        'plots': base_path / "plots",
        'results': base_path / "results",
        'logs': base_path / "logs"
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def build_mlp(hidden_units: int = 64, input_dim: int = 784,
              initializer: str = 'glorot', optimizer: str = 'adam') -> tf.keras.Model:
    """Constrói um MLP simples com 1 camada oculta.

    Args:
        hidden_units: número de neurônios na camada oculta
        input_dim: dimensão da entrada
        initializer: tipo de inicialização ('glorot', 'he', 'normal', 'constant')
        optimizer: tipo de otimizador ('adam', 'sgd')

    Returns:
        modelo compilado
    """
    # Selecionar inicializador
    init_options = {
        'glorot': GlorotUniform(seed=CONFIG.seed),
        'he': HeUniform(seed=CONFIG.seed),
        'normal': RandomNormal(mean=0.0, stddev=0.05, seed=CONFIG.seed),
        'constant': Constant(value=0.05)
    }

    if initializer not in init_options:
        raise ValueError(f'Initializer deve ser um de: {list(init_options.keys())}')

    init = init_options[initializer]

    # Selecionar otimizador
    if optimizer == 'adam':
        opt = Adam(learning_rate=CONFIG.learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=CONFIG.learning_rate, momentum=0.9)
    else:
        raise ValueError('Optimizer deve ser "adam" ou "sgd"')

    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(hidden_units, activation='relu',
              kernel_initializer=init,
              bias_initializer=Constant(0.0),
              name='hidden_layer'),
        Dense(10, activation='softmax',
              kernel_initializer=init,
              bias_initializer=Constant(0.0),
              name='output_layer')
    ])

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logger.info(f"Modelo construído - Initializer: {initializer}, Optimizer: {optimizer}")
    return model


def run_experiment(initializer_name: str = 'glorot', optimizer: str = 'adam',
                  epochs: int = None, noise_std: float = None,
                  run_name: str = None, save_model: bool = True,
                  dirs: Dict[str, Path] = None) -> Dict:
    """Executa um experimento completo com um tipo de inicialização.

    Args:
        initializer_name: tipo de inicialização
        optimizer: tipo de otimizador
        epochs: número de épocas
        noise_std: desvio padrão do ruído
        run_name: nome do experimento
        save_model: se deve salvar o modelo
        dirs: diretórios para salvar resultados

    Returns:
        dict: resultados do experimento
    """
    # Usar configurações padrão se não especificadas
    epochs = epochs or CONFIG.epochs
    noise_std = noise_std or CONFIG.noise_std
    run_name = run_name or f"{initializer_name}_{optimizer}"

    logger.info(f"Iniciando experimento: {run_name}")
    logger.info(f"Configurações - Initializer: {initializer_name}, Optimizer: {optimizer}")
    logger.info(f"Épocas: {epochs}, Ruído: {noise_std}")

    start_time = time.time()

    try:
        # Carregar MNIST
        logger.info("Carregando dataset MNIST...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Pré-processar dados
        logger.info("Pré-processando dados...")
        x_train, x_test = preprocess_mnist(x_train, x_test)

        # Criar versão ruidosa do teste
        x_test_noisy = add_gaussian_noise(x_test, std=noise_std, seed=CONFIG.seed)

        # Construir modelo
        model = build_mlp(hidden_units=CONFIG.hidden_units,
                         initializer=initializer_name,
                         optimizer=optimizer)

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=CONFIG.patience,
                         restore_best_weights=True, min_delta=CONFIG.min_delta),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]

        if save_model and dirs:
            model_path = dirs['models'] / f"{run_name}_best.h5"
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True))

        # Treinar modelo
        logger.info("Iniciando treinamento...")
        history = model.fit(
            x_train, y_train,
            batch_size=CONFIG.batch_size,
            epochs=epochs,
            validation_split=CONFIG.validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Avaliar em dados limpos
        logger.info("Avaliando em dados limpos...")
        clean_results = evaluate_model_performance(model, x_test, y_test, verbose=0)

        # Avaliar em dados ruidosos
        logger.info("Avaliando em dados ruidosos...")
        noisy_results = evaluate_model_performance(model, x_test_noisy, y_test, verbose=0)

        # Calcular métricas adicionais
        training_time = time.time() - start_time
        degradation = (clean_results['test_accuracy'] - noisy_results['test_accuracy']) * 100

        # Preparar resultados
        results = {
            'run_name': run_name,
            'initializer': initializer_name,
            'optimizer': optimizer,
            'config': CONFIG.to_dict(),
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'clean_results': clean_results,
            'noisy_results': noisy_results,
            'degradation': degradation,
            'history': history.history
        }

        # Log dos resultados
        logger.info(f"Experimento concluído em {training_time:.2f}s")
        logger.info(f"Dados limpos - Loss: {clean_results['test_loss']:.4f}, Accuracy: {clean_results['test_accuracy']:.4f}")
        logger.info(f"Dados ruidosos - Loss: {noisy_results['test_loss']:.4f}, Accuracy: {noisy_results['test_accuracy']:.4f}")
        logger.info(f"Degradação accuracy: {degradation:.2f}%")

        return results

    except Exception as e:
        logger.error(f"Erro no experimento {run_name}: {str(e)}")
        raise


def save_results(results: List[Dict], dirs: Dict[str, Path]):
    """Salva os resultados dos experimentos em arquivos."""
    # Salvar resultados em JSON
    results_file = dirs['results'] / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Criar relatório resumido
    report_file = dirs['results'] / "summary_report.txt"
    with open(report_file, 'w') as f:
        f.write("RELATÓRIO DE EXPERIMENTOS - COMPARAÇÃO DE INICIALIZAÇÕES\n")
        f.write("=" * 60 + "\n\n")

        for result in results:
            f.write(f"Experimento: {result['run_name']}\n")
            f.write(f"Initializer: {result['initializer']}\n")
            f.write(f"Optimizer: {result['optimizer']}\n")
            f.write(f"Tempo de treinamento: {result['training_time']:.2f}s\n")
            f.write(f"Épocas treinadas: {result['epochs_trained']}\n")
            f.write(f"Accuracy (limpo): {result['clean_results']['test_accuracy']:.4f}\n")
            f.write(f"Accuracy (ruidoso): {result['noisy_results']['test_accuracy']:.4f}\n")
            f.write(f"Degradação: {result['degradation']:.2f}%\n")
            f.write("-" * 40 + "\n\n")

    logger.info(f"Resultados salvos em: {dirs['results']}")


def visualize_results(results: List[Dict], dirs: Dict[str, Path]):
    """Cria visualizações dos resultados."""
    # Preparar dados para plotagem
    histories = {}
    for result in results:
        histories[result['run_name']] = type('obj', (object,), {'history': result['history']})

    # Salvar gráfico de comparação
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8')

    plot_history(histories, title="Comparação de Inicializações")
    plt.savefig(dirs['plots'] / "training_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico de barras com métricas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    names = [r['run_name'] for r in results]
    clean_acc = [r['clean_results']['test_accuracy'] for r in results]
    noisy_acc = [r['noisy_results']['test_accuracy'] for r in results]
    degradation = [r['degradation'] for r in results]

    # Accuracy comparison
    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width/2, clean_acc, width, label='Dados Limpos', alpha=0.8)
    ax1.bar(x + width/2, noisy_acc, width, label='Dados Ruidosos', alpha=0.8)
    ax1.set_xlabel('Experimentos')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Comparação de Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Degradation
    ax2.bar(names, degradation, alpha=0.8, color='red')
    ax2.set_xlabel('Experimentos')
    ax2.set_ylabel('Degradação (%)')
    ax2.set_title('Degradação por Ruído')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(dirs['plots'] / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizações salvas em: {dirs['plots']}")


def main():
    """Função principal que executa todos os experimentos."""
    logger.info("Iniciando experimentos de comparação de inicializações...")

    # Configurar diretórios
    dirs = setup_experiment_directories()
    logger.info(f"Diretórios criados em: {dirs['base']}")

    # Lista de experimentos expandida
    experiments = [
        {'initializer': 'glorot', 'optimizer': 'adam', 'run_name': 'GlorotUniform_Adam'},
        {'initializer': 'he', 'optimizer': 'adam', 'run_name': 'HeUniform_Adam'},
        {'initializer': 'normal', 'optimizer': 'adam', 'run_name': 'RandomNormal_Adam'},
        {'initializer': 'constant', 'optimizer': 'adam', 'run_name': 'Constant_Adam'},
        {'initializer': 'glorot', 'optimizer': 'sgd', 'run_name': 'GlorotUniform_SGD'},
        {'initializer': 'he', 'optimizer': 'sgd', 'run_name': 'HeUniform_SGD'},
    ]

    # Executar experimentos
    all_results = []
    histories = {}

    for i, exp in enumerate(experiments, 1):
        logger.info(f"Executando experimento {i}/{len(experiments)}")

        result = run_experiment(
            initializer_name=exp['initializer'],
            optimizer=exp['optimizer'],
            epochs=CONFIG.epochs,
            noise_std=CONFIG.noise_std,
            run_name=exp['run_name'],
            save_model=True,
            dirs=dirs
        )

        all_results.append(result)
        histories[exp['run_name']] = type('obj', (object,), {'history': result['history']})

    # Salvar resultados
    save_results(all_results, dirs)

    # Criar visualizações
    visualize_results(all_results, dirs)

    # Resumo final
    logger.info("\n" + "="*60)
    logger.info("RESUMO FINAL DOS EXPERIMENTOS")
    logger.info("="*60)

    best_clean = max(all_results, key=lambda x: x['clean_results']['test_accuracy'])
    best_robust = min(all_results, key=lambda x: x['degradation'])

    logger.info(f"Melhor accuracy (limpo): {best_clean['run_name']} - {best_clean['clean_results']['test_accuracy']:.4f}")
    logger.info(f"Mais robusto ao ruído: {best_robust['run_name']} - degradação: {best_robust['degradation']:.2f}%")
    logger.info(f"Todos os resultados salvos em: {dirs['base']}")
    logger.info("Experimentos concluídos com sucesso!")


if __name__ == "__main__":
    main()