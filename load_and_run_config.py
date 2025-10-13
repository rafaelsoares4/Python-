# load_and_run_config.py
# Script para executar experimentos a partir de configurações JSON pré-definidas

import json
import argparse
import sys
from pathlib import Path
from advanced_experiments import (
    setup_experiment_directories,
    run_advanced_experiment,
    logger
)


def load_config(config_name: str, config_file: str = "experiment_configs.json") -> dict:
    """Carrega uma configuração do arquivo JSON.

    Args:
        config_name: nome da configuração
        config_file: arquivo JSON com configurações

    Returns:
        dict: configuração carregada
    """
    config_path = Path(config_file)

    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_file}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    # Remover comentários
    all_configs = {k: v for k, v in all_configs.items() if not k.startswith('_')}

    if config_name not in all_configs:
        logger.error(f"Configuração '{config_name}' não encontrada.")
        logger.info(f"Configurações disponíveis: {', '.join(all_configs.keys())}")
        sys.exit(1)

    return all_configs[config_name]


def list_configs(config_file: str = "experiment_configs.json"):
    """Lista todas as configurações disponíveis."""
    config_path = Path(config_file)

    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_file}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    # Remover comentários
    all_configs = {k: v for k, v in all_configs.items() if not k.startswith('_')}

    print("\n" + "="*80)
    print("CONFIGURAÇÕES DISPONÍVEIS")
    print("="*80 + "\n")

    for name, config in all_configs.items():
        print(f"📌 {name}")
        print(f"   Descrição: {config.get('description', 'N/A')}")
        print(f"   Inicializador: {config.get('initializer', 'N/A')}")
        print(f"   Otimizador: {config.get('optimizer', 'N/A')}")
        print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"   Dropout: {config.get('dropout_rate', 'N/A')}")
        print(f"   Ruído: {config.get('noise_std', 'N/A')}")
        print()


def run_from_config(config_name: str, config_file: str = "experiment_configs.json"):
    """Executa um experimento a partir de uma configuração pré-definida.

    Args:
        config_name: nome da configuração
        config_file: arquivo JSON com configurações
    """
    logger.info(f"Carregando configuração: {config_name}")
    config = load_config(config_name, config_file)

    logger.info(f"Descrição: {config.get('description', 'N/A')}\n")

    # Criar diretórios
    dirs = setup_experiment_directories()
    logger.info(f"Diretórios criados em: {dirs['base']}\n")

    # Preparar configuração completa
    full_config = {
        'run_name': config.get('name', config_name),
        'initializer': config.get('initializer', 'he'),
        'optimizer': config.get('optimizer', 'adam'),
        'learning_rate': config.get('learning_rate', 1e-3),
        'hidden_units': config.get('hidden_units', 64),
        'batch_size': config.get('batch_size', 128),
        'epochs': config.get('epochs', 15),
        'noise_std': config.get('noise_std', 0.2),
        'dropout_rate': config.get('dropout_rate', 0.0),
        'activation': config.get('activation', 'relu'),
        'regularizer': config.get('regularizer', None),
        'reg_strength': config.get('reg_strength', 0.01),
        'seed': config.get('seed', 42),
        'patience': config.get('patience', 3),
        'validation_split': config.get('validation_split', 0.1),
        'save_model': config.get('save_model', True)
    }

    # Executar experimento
    result = run_advanced_experiment(full_config, dirs)

    # Exibir resumo
    logger.info("\n" + "="*80)
    logger.info("RESUMO DOS RESULTADOS")
    logger.info("="*80)
    logger.info(f"Configuração:             {config_name}")
    logger.info(f"Accuracy (limpo):         {result['clean_accuracy']:.4f}")
    logger.info(f"Accuracy (ruidoso):       {result['noisy_accuracy']:.4f}")
    logger.info(f"Degradação:               {result['degradation']:.2f}%")
    logger.info(f"Tempo de treinamento:     {result['training_time']:.2f}s")
    logger.info(f"Épocas treinadas:         {result['epochs_trained']}")
    logger.info(f"Resultados salvos em:     {dirs['base']}")
    logger.info("="*80 + "\n")


def run_multiple_configs(config_names: list, config_file: str = "experiment_configs.json"):
    """Executa múltiplos experimentos sequencialmente.

    Args:
        config_names: lista de nomes de configurações
        config_file: arquivo JSON com configurações
    """
    logger.info(f"Executando {len(config_names)} experimentos...")

    # Criar diretórios
    dirs = setup_experiment_directories()
    logger.info(f"Diretórios criados em: {dirs['base']}\n")

    all_results = []

    for i, config_name in enumerate(config_names, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENTO {i}/{len(config_names)}: {config_name}")
        logger.info('='*80 + "\n")

        config = load_config(config_name, config_file)

        full_config = {
            'run_name': config.get('name', config_name),
            'initializer': config.get('initializer', 'he'),
            'optimizer': config.get('optimizer', 'adam'),
            'learning_rate': config.get('learning_rate', 1e-3),
            'hidden_units': config.get('hidden_units', 64),
            'batch_size': config.get('batch_size', 128),
            'epochs': config.get('epochs', 15),
            'noise_std': config.get('noise_std', 0.2),
            'dropout_rate': config.get('dropout_rate', 0.0),
            'activation': config.get('activation', 'relu'),
            'regularizer': config.get('regularizer', None),
            'reg_strength': config.get('reg_strength', 0.01),
            'seed': config.get('seed', 42),
            'patience': config.get('patience', 3),
            'validation_split': config.get('validation_split', 0.1),
            'save_model': False  # Não salvar modelos para economizar espaço
        }

        result = run_advanced_experiment(full_config, dirs)
        all_results.append(result)

    # Resumo final
    logger.info("\n" + "="*80)
    logger.info("RESUMO COMPARATIVO")
    logger.info("="*80 + "\n")

    for i, (name, result) in enumerate(zip(config_names, all_results), 1):
        logger.info(f"{i}. {name}:")
        logger.info(f"   Clean Acc: {result['clean_accuracy']:.4f} | "
                   f"Noisy Acc: {result['noisy_accuracy']:.4f} | "
                   f"Degradação: {result['degradation']:.2f}%")

    best = max(all_results, key=lambda x: x['clean_accuracy'])
    most_robust = min(all_results, key=lambda x: x['degradation'])

    logger.info(f"\n✨ Melhor accuracy: {best['run_name']} ({best['clean_accuracy']:.4f})")
    logger.info(f"🛡️  Mais robusto: {most_robust['run_name']} (degradação: {most_robust['degradation']:.2f}%)")
    logger.info(f"\n📁 Resultados salvos em: {dirs['base']}")
    logger.info("="*80 + "\n")


def main():
    """Função principal com interface de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Executar experimentos a partir de configurações pré-definidas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Listar todas as configurações disponíveis
  python load_and_run_config.py --list

  # Executar uma configuração específica
  python load_and_run_config.py --config baseline

  # Executar múltiplas configurações
  python load_and_run_config.py --configs baseline high_robustness fast_training

  # Usar arquivo de configuração customizado
  python load_and_run_config.py --config baseline --file my_configs.json
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true',
                      help='Listar todas as configurações disponíveis')
    group.add_argument('--config', type=str,
                      help='Nome da configuração a executar')
    group.add_argument('--configs', nargs='+',
                      help='Lista de configurações a executar')

    parser.add_argument('--file', type=str, default='experiment_configs.json',
                       help='Arquivo JSON com configurações (default: experiment_configs.json)')

    args = parser.parse_args()

    if args.list:
        list_configs(args.file)
    elif args.config:
        run_from_config(args.config, args.file)
    elif args.configs:
        run_multiple_configs(args.configs, args.file)


if __name__ == "__main__":
    main()

