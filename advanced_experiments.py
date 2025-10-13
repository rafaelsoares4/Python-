# advanced_experiments.py
# Script para executar experimentos avançados e sistemáticos com variação de hiperparâmetros

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.initializers import RandomNormal, Constant, GlorotUniform, HeUniform
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2

from utils import preprocess_mnist, add_gaussian_noise
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_experiment_directories(base_dir: str = "outputs_advanced") -> Dict[str, Path]:
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


def build_mlp_advanced(hidden_units: int = 64,
                       input_dim: int = 784,
                       initializer: str = 'glorot',
                       optimizer: str = 'adam',
                       learning_rate: float = 1e-3,
                       dropout_rate: float = 0.0,
                       activation: str = 'relu',
                       regularizer: str = None,
                       reg_strength: float = 0.01,
                       seed: int = 42) -> tf.keras.Model:
    """Constrói um MLP com opções avançadas.

    Args:
        hidden_units: número de neurônios na camada oculta
        input_dim: dimensão da entrada
        initializer: tipo de inicialização ('glorot', 'he', 'normal', 'constant')
        optimizer: tipo de otimizador ('adam', 'sgd')
        learning_rate: taxa de aprendizado
        dropout_rate: taxa de dropout (0.0 = sem dropout)
        activation: função de ativação ('relu', 'tanh', 'elu', 'sigmoid')
        regularizer: tipo de regularização ('l1', 'l2', 'l1_l2', None)
        reg_strength: força da regularização
        seed: seed para reprodutibilidade

    Returns:
        modelo compilado
    """
    # Selecionar inicializador
    init_options = {
        'glorot': GlorotUniform(seed=seed),
        'he': HeUniform(seed=seed),
        'normal': RandomNormal(mean=0.0, stddev=0.05, seed=seed),
        'constant': Constant(value=0.05)
    }

    if initializer not in init_options:
        raise ValueError(f'Initializer deve ser um de: {list(init_options.keys())}')

    init = init_options[initializer]

    # Selecionar regularizador
    reg = None
    if regularizer == 'l1':
        reg = l1(reg_strength)
    elif regularizer == 'l2':
        reg = l2(reg_strength)
    elif regularizer == 'l1_l2':
        reg = l1_l2(l1=reg_strength/2, l2=reg_strength/2)

    # Selecionar otimizador
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError('Optimizer deve ser "adam" ou "sgd"')

    # Construir modelo
    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(hidden_units,
              activation=activation,
              kernel_initializer=init,
              bias_initializer=Constant(0.0),
              kernel_regularizer=reg,
              name='hidden_layer'),
    ])

    # Adicionar dropout se especificado
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate, seed=seed))

    model.add(Dense(10,
                    activation='softmax',
                    kernel_initializer=init,
                    bias_initializer=Constant(0.0),
                    name='output_layer'))

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logger.info(f"Modelo construído - Init: {initializer}, Opt: {optimizer}, LR: {learning_rate}, "
                f"Dropout: {dropout_rate}, Activation: {activation}, Reg: {regularizer}")

    return model


def run_advanced_experiment(config: Dict, dirs: Dict[str, Path] = None) -> Dict:
    """Executa um experimento com configurações personalizadas.

    Args:
        config: dicionário com todas as configurações do experimento
        dirs: diretórios para salvar resultados

    Returns:
        dict: resultados do experimento
    """
    run_name = config['run_name']
    logger.info(f"Iniciando experimento: {run_name}")
    logger.info(f"Configurações: {config}")

    start_time = time.time()

    try:
        # Carregar MNIST
        logger.info("Carregando dataset MNIST...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Pré-processar dados
        logger.info("Pré-processando dados...")
        x_train, x_test = preprocess_mnist(x_train, x_test)

        # Criar versão ruidosa do teste
        noise_std = config.get('noise_std', 0.2)
        x_test_noisy = add_gaussian_noise(x_test, std=noise_std, seed=config.get('seed', 42))

        # Construir modelo
        model = build_mlp_advanced(
            hidden_units=config.get('hidden_units', 64),
            initializer=config.get('initializer', 'glorot'),
            optimizer=config.get('optimizer', 'adam'),
            learning_rate=config.get('learning_rate', 1e-3),
            dropout_rate=config.get('dropout_rate', 0.0),
            activation=config.get('activation', 'relu'),
            regularizer=config.get('regularizer', None),
            reg_strength=config.get('reg_strength', 0.01),
            seed=config.get('seed', 42)
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss',
                         patience=config.get('patience', 3),
                         restore_best_weights=True,
                         min_delta=config.get('min_delta', 0.001)),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]

        if config.get('save_model', True) and dirs:
            model_path = dirs['models'] / f"{run_name}_best.h5"
            callbacks.append(ModelCheckpoint(str(model_path), save_best_only=True))

        # Treinar modelo
        logger.info("Iniciando treinamento...")
        history = model.fit(
            x_train, y_train,
            batch_size=config.get('batch_size', 128),
            epochs=config.get('epochs', 15),
            validation_split=config.get('validation_split', 0.1),
            callbacks=callbacks,
            verbose=1
        )

        # Avaliar em dados limpos
        logger.info("Avaliando em dados limpos...")
        clean_loss, clean_acc = model.evaluate(x_test, y_test, verbose=0)

        # Avaliar em dados ruidosos
        logger.info("Avaliando em dados ruidosos...")
        noisy_loss, noisy_acc = model.evaluate(x_test_noisy, y_test, verbose=0)

        # Calcular métricas adicionais
        training_time = time.time() - start_time
        degradation = (clean_acc - noisy_acc) * 100

        # Preparar resultados
        results = {
            'run_name': run_name,
            'config': config,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'clean_loss': float(clean_loss),
            'clean_accuracy': float(clean_acc),
            'noisy_loss': float(noisy_loss),
            'noisy_accuracy': float(noisy_acc),
            'degradation': float(degradation),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }

        # Log dos resultados
        logger.info(f"Experimento concluído em {training_time:.2f}s")
        logger.info(f"Dados limpos - Loss: {clean_loss:.4f}, Accuracy: {clean_acc:.4f}")
        logger.info(f"Dados ruidosos - Loss: {noisy_loss:.4f}, Accuracy: {noisy_acc:.4f}")
        logger.info(f"Degradação accuracy: {degradation:.2f}%")

        return results

    except Exception as e:
        logger.error(f"Erro no experimento {run_name}: {str(e)}")
        raise


def save_results(results: List[Dict], dirs: Dict[str, Path], experiment_type: str):
    """Salva os resultados dos experimentos em arquivos."""
    # Salvar resultados em JSON
    results_file = dirs['results'] / f"{experiment_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Criar relatório resumido
    report_file = dirs['results'] / f"{experiment_type}_summary.txt"
    with open(report_file, 'w') as f:
        f.write(f"RELATÓRIO DE EXPERIMENTOS - {experiment_type.upper()}\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Experimento: {result['run_name']}\n")
            f.write(f"Configurações:\n")
            for key, value in result['config'].items():
                if key != 'run_name':
                    f.write(f"  {key}: {value}\n")
            f.write(f"Tempo de treinamento: {result['training_time']:.2f}s\n")
            f.write(f"Épocas treinadas: {result['epochs_trained']}\n")
            f.write(f"Accuracy (limpo): {result['clean_accuracy']:.4f}\n")
            f.write(f"Accuracy (ruidoso): {result['noisy_accuracy']:.4f}\n")
            f.write(f"Degradação: {result['degradation']:.2f}%\n")
            f.write("-" * 60 + "\n\n")

    logger.info(f"Resultados salvos em: {dirs['results']}")


def create_comparison_plots(results: List[Dict], dirs: Dict[str, Path], 
                           experiment_type: str, x_param: str, x_label: str):
    """Cria gráficos de comparação para experimentos."""
    plt.style.use('seaborn-v0_8')

    # Extrair dados
    x_values = [r['config'][x_param] for r in results]
    clean_acc = [r['clean_accuracy'] for r in results]
    noisy_acc = [r['noisy_accuracy'] for r in results]
    degradation = [r['degradation'] for r in results]
    training_time = [r['training_time'] for r in results]

    # Criar figura com múltiplos subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy vs parâmetro
    axes[0, 0].plot(x_values, clean_acc, 'o-', label='Dados Limpos', linewidth=2, markersize=8)
    axes[0, 0].plot(x_values, noisy_acc, 's-', label='Dados Ruidosos', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel(x_label, fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title(f'Accuracy vs {x_label}', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Degradação vs parâmetro
    axes[0, 1].plot(x_values, degradation, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel(x_label, fontsize=12)
    axes[0, 1].set_ylabel('Degradação (%)', fontsize=12)
    axes[0, 1].set_title(f'Degradação por Ruído vs {x_label}', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Tempo de treinamento vs parâmetro
    axes[1, 0].plot(x_values, training_time, 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel(x_label, fontsize=12)
    axes[1, 0].set_ylabel('Tempo (s)', fontsize=12)
    axes[1, 0].set_title(f'Tempo de Treinamento vs {x_label}', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Tabela de resultados
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')

    table_data = []
    headers = [x_label, 'Clean Acc', 'Noisy Acc', 'Degradação']
    
    for i in range(len(results)):
        row = [
            str(x_values[i]),
            f"{clean_acc[i]:.4f}",
            f"{noisy_acc[i]:.4f}",
            f"{degradation[i]:.2f}%"
        ]
        table_data.append(row)

    table = axes[1, 1].table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Colorir cabeçalho
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(dirs['plots'] / f"{experiment_type}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Gráficos salvos em: {dirs['plots']}")


# ========== EXPERIMENTOS ESPECÍFICOS ==========

def experiment_noise_levels(dirs: Dict[str, Path]):
    """Experimento 1: Testar diferentes níveis de ruído."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTO 1: VARIAÇÃO DE NÍVEIS DE RUÍDO")
    logger.info("="*80 + "\n")

    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    results = []

    for noise in noise_levels:
        config = {
            'run_name': f'noise_{noise:.2f}',
            'initializer': 'he',
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'hidden_units': 64,
            'batch_size': 128,
            'epochs': 15,
            'noise_std': noise,
            'dropout_rate': 0.0,
            'activation': 'relu',
            'seed': 42,
            'patience': 3,
            'validation_split': 0.1,
            'save_model': False
        }

        result = run_advanced_experiment(config, dirs)
        results.append(result)

    save_results(results, dirs, 'noise_levels')
    create_comparison_plots(results, dirs, 'noise_levels', 'noise_std', 'Desvio Padrão do Ruído')

    return results


def experiment_learning_rates(dirs: Dict[str, Path]):
    """Experimento 2: Testar diferentes taxas de aprendizado."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTO 2: VARIAÇÃO DE TAXAS DE APRENDIZADO")
    logger.info("="*80 + "\n")

    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    results = []

    for lr in learning_rates:
        config = {
            'run_name': f'lr_{lr:.0e}',
            'initializer': 'he',
            'optimizer': 'adam',
            'learning_rate': lr,
            'hidden_units': 64,
            'batch_size': 128,
            'epochs': 15,
            'noise_std': 0.2,
            'dropout_rate': 0.0,
            'activation': 'relu',
            'seed': 42,
            'patience': 3,
            'validation_split': 0.1,
            'save_model': False
        }

        result = run_advanced_experiment(config, dirs)
        results.append(result)

    save_results(results, dirs, 'learning_rates')
    create_comparison_plots(results, dirs, 'learning_rates', 'learning_rate', 'Taxa de Aprendizado (log scale)')

    return results


def experiment_batch_sizes(dirs: Dict[str, Path]):
    """Experimento 3: Testar diferentes tamanhos de batch."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTO 3: VARIAÇÃO DE TAMANHOS DE BATCH")
    logger.info("="*80 + "\n")

    batch_sizes = [32, 64, 128, 256, 512]
    results = []

    for batch_size in batch_sizes:
        config = {
            'run_name': f'batch_{batch_size}',
            'initializer': 'he',
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'hidden_units': 64,
            'batch_size': batch_size,
            'epochs': 15,
            'noise_std': 0.2,
            'dropout_rate': 0.0,
            'activation': 'relu',
            'seed': 42,
            'patience': 3,
            'validation_split': 0.1,
            'save_model': False
        }

        result = run_advanced_experiment(config, dirs)
        results.append(result)

    save_results(results, dirs, 'batch_sizes')
    create_comparison_plots(results, dirs, 'batch_sizes', 'batch_size', 'Tamanho do Batch')

    return results


def experiment_dropout(dirs: Dict[str, Path]):
    """Experimento 4: Testar diferentes taxas de dropout."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTO 4: VARIAÇÃO DE TAXAS DE DROPOUT")
    logger.info("="*80 + "\n")

    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for dropout in dropout_rates:
        config = {
            'run_name': f'dropout_{dropout:.1f}',
            'initializer': 'he',
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'hidden_units': 64,
            'batch_size': 128,
            'epochs': 15,
            'noise_std': 0.2,
            'dropout_rate': dropout,
            'activation': 'relu',
            'seed': 42,
            'patience': 3,
            'validation_split': 0.1,
            'save_model': False
        }

        result = run_advanced_experiment(config, dirs)
        results.append(result)

    save_results(results, dirs, 'dropout_rates')
    create_comparison_plots(results, dirs, 'dropout_rates', 'dropout_rate', 'Taxa de Dropout')

    return results


def experiment_activations(dirs: Dict[str, Path]):
    """Experimento 5: Testar diferentes funções de ativação."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTO 5: VARIAÇÃO DE FUNÇÕES DE ATIVAÇÃO")
    logger.info("="*80 + "\n")

    activations = ['relu', 'tanh', 'elu', 'sigmoid']
    results = []

    for activation in activations:
        config = {
            'run_name': f'activation_{activation}',
            'initializer': 'he' if activation in ['relu', 'elu'] else 'glorot',
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'hidden_units': 64,
            'batch_size': 128,
            'epochs': 15,
            'noise_std': 0.2,
            'dropout_rate': 0.0,
            'activation': activation,
            'seed': 42,
            'patience': 3,
            'validation_split': 0.1,
            'save_model': False
        }

        result = run_advanced_experiment(config, dirs)
        results.append(result)

    save_results(results, dirs, 'activations')

    # Criar gráfico específico para ativações
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    names = [r['run_name'] for r in results]
    clean_acc = [r['clean_accuracy'] for r in results]
    noisy_acc = [r['noisy_accuracy'] for r in results]
    degradation = [r['degradation'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    # Accuracy comparison
    axes[0].bar(x - width/2, clean_acc, width, label='Dados Limpos', alpha=0.8)
    axes[0].bar(x + width/2, noisy_acc, width, label='Dados Ruidosos', alpha=0.8)
    axes[0].set_xlabel('Função de Ativação', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Comparação de Accuracy por Ativação', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(activations)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Degradation
    axes[1].bar(activations, degradation, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[1].set_xlabel('Função de Ativação', fontsize=12)
    axes[1].set_ylabel('Degradação (%)', fontsize=12)
    axes[1].set_title('Degradação por Ruído', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(dirs['plots'] / 'activations_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


def main():
    """Função principal que executa todos os experimentos avançados."""
    logger.info("\n" + "="*80)
    logger.info("INICIANDO EXPERIMENTOS AVANÇADOS - MLP 64 NEURÔNIOS")
    logger.info("="*80 + "\n")

    # Configurar diretórios
    dirs = setup_experiment_directories()
    logger.info(f"Diretórios criados em: {dirs['base']}\n")

    # Executar todos os experimentos
    all_results = {}

    all_results['noise_levels'] = experiment_noise_levels(dirs)
    all_results['learning_rates'] = experiment_learning_rates(dirs)
    all_results['batch_sizes'] = experiment_batch_sizes(dirs)
    all_results['dropout_rates'] = experiment_dropout(dirs)
    all_results['activations'] = experiment_activations(dirs)

    # Salvar resumo geral
    summary_file = dirs['results'] / "overall_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RESUMO GERAL DE TODOS OS EXPERIMENTOS\n")
        f.write("=" * 80 + "\n\n")

        for exp_type, results in all_results.items():
            f.write(f"\n{exp_type.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")

            best = max(results, key=lambda x: x['clean_accuracy'])
            most_robust = min(results, key=lambda x: x['degradation'])

            f.write(f"Melhor accuracy (limpo): {best['run_name']} - {best['clean_accuracy']:.4f}\n")
            f.write(f"Mais robusto ao ruído: {most_robust['run_name']} - degradação: {most_robust['degradation']:.2f}%\n")
            f.write(f"Total de experimentos: {len(results)}\n\n")

    logger.info("\n" + "="*80)
    logger.info("TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO!")
    logger.info(f"Resultados salvos em: {dirs['base']}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

