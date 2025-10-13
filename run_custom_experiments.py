# run_custom_experiments.py
# Script para executar experimentos personalizados de forma flexível

import argparse
from advanced_experiments import (
    setup_experiment_directories,
    experiment_noise_levels,
    experiment_learning_rates,
    experiment_batch_sizes,
    experiment_dropout,
    experiment_activations,
    run_advanced_experiment,
    logger
)


def run_specific_experiments(experiments_to_run: list):
    """Executa apenas os experimentos especificados.

    Args:
        experiments_to_run: lista com nomes dos experimentos a executar
    """
    # Configurar diretórios
    dirs = setup_experiment_directories()
    logger.info(f"Diretórios criados em: {dirs['base']}\n")

    # Mapear nomes de experimentos para funções
    experiment_map = {
        'noise': experiment_noise_levels,
        'learning_rate': experiment_learning_rates,
        'batch_size': experiment_batch_sizes,
        'dropout': experiment_dropout,
        'activation': experiment_activations
    }

    results = {}

    for exp_name in experiments_to_run:
        if exp_name in experiment_map:
            logger.info(f"\nExecutando experimento: {exp_name}")
            results[exp_name] = experiment_map[exp_name](dirs)
        else:
            logger.warning(f"Experimento '{exp_name}' não reconhecido. Ignorando...")

    logger.info("\n" + "="*80)
    logger.info(f"Experimentos concluídos! Resultados salvos em: {dirs['base']}")
    logger.info("="*80 + "\n")

    return results


def run_single_custom_experiment(config: dict):
    """Executa um único experimento com configurações customizadas.

    Args:
        config: dicionário com configurações do experimento
    """
    dirs = setup_experiment_directories()
    logger.info(f"Diretórios criados em: {dirs['base']}\n")

    result = run_advanced_experiment(config, dirs)

    logger.info("\n" + "="*80)
    logger.info("Experimento concluído!")
    logger.info(f"Resultados salvos em: {dirs['base']}")
    logger.info("="*80 + "\n")

    return result


def main():
    """Função principal com interface de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Executar experimentos customizados no MLP MNIST',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Executar todos os experimentos
  python run_custom_experiments.py --all

  # Executar experimentos específicos
  python run_custom_experiments.py --experiments noise learning_rate

  # Executar experimento customizado
  python run_custom_experiments.py --custom \\
    --initializer he \\
    --optimizer adam \\
    --learning-rate 0.001 \\
    --dropout 0.2 \\
    --noise 0.3

Experimentos disponíveis:
  - noise: Testa diferentes níveis de ruído
  - learning_rate: Testa diferentes taxas de aprendizado
  - batch_size: Testa diferentes tamanhos de batch
  - dropout: Testa diferentes taxas de dropout
  - activation: Testa diferentes funções de ativação
        """
    )

    # Grupos de argumentos
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true',
                      help='Executar todos os experimentos')
    group.add_argument('--experiments', nargs='+',
                      choices=['noise', 'learning_rate', 'batch_size', 'dropout', 'activation'],
                      help='Executar experimentos específicos')
    group.add_argument('--custom', action='store_true',
                      help='Executar um experimento customizado')

    # Argumentos para experimento customizado
    parser.add_argument('--initializer', type=str, default='he',
                       choices=['glorot', 'he', 'normal', 'constant'],
                       help='Tipo de inicialização (default: he)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Otimizador (default: adam)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Taxa de aprendizado (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Tamanho do batch (default: 128)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Número de épocas (default: 15)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Taxa de dropout (default: 0.0)')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'elu', 'sigmoid'],
                       help='Função de ativação (default: relu)')
    parser.add_argument('--noise', type=float, default=0.2,
                       help='Desvio padrão do ruído (default: 0.2)')
    parser.add_argument('--hidden-units', type=int, default=64,
                       help='Neurônios na camada oculta (default: 64)')
    parser.add_argument('--name', type=str, default='custom_experiment',
                       help='Nome do experimento (default: custom_experiment)')

    args = parser.parse_args()

    # Executar experimentos baseado nos argumentos
    if args.all:
        logger.info("Executando TODOS os experimentos...")
        experiments = ['noise', 'learning_rate', 'batch_size', 'dropout', 'activation']
        run_specific_experiments(experiments)

    elif args.experiments:
        logger.info(f"Executando experimentos selecionados: {args.experiments}")
        run_specific_experiments(args.experiments)

    elif args.custom:
        logger.info("Executando experimento customizado...")
        config = {
            'run_name': args.name,
            'initializer': args.initializer,
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'hidden_units': args.hidden_units,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'noise_std': args.noise,
            'dropout_rate': args.dropout,
            'activation': args.activation,
            'seed': 42,
            'patience': 3,
            'validation_split': 0.1,
            'save_model': True
        }
        run_single_custom_experiment(config)


if __name__ == "__main__":
    main()

