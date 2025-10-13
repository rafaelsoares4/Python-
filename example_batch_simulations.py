#!/usr/bin/env python3
# example_batch_simulations.py
# Exemplo de como executar múltiplas simulações de forma programática

"""
Este script demonstra como executar múltiplas simulações
personalizadas de forma programática.

Use este como template para criar seus próprios experimentos em batch.
"""

from advanced_experiments import (
    setup_experiment_directories,
    run_advanced_experiment,
    save_results,
    logger
)
import matplotlib.pyplot as plt
import numpy as np


def example_1_compare_noise_with_dropout():
    """Exemplo 1: Comparar robustez ao ruído com e sem dropout."""

    logger.info("\n" + "="*80)
    logger.info("EXEMPLO 1: RUÍDO COM E SEM DROPOUT")
    logger.info("="*80 + "\n")

    dirs = setup_experiment_directories(base_dir="outputs_example_1")

    # Configurações base
    base_config = {
        'initializer': 'he',
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'hidden_units': 64,
        'batch_size': 128,
        'epochs': 15,
        'activation': 'relu',
        'seed': 42,
        'patience': 3,
        'validation_split': 0.1,
        'save_model': False
    }

    # Experimentos: variar ruído e dropout
    experiments = []

    for noise in [0.1, 0.2, 0.3]:
        for dropout in [0.0, 0.2, 0.4]:
            config = base_config.copy()
            config['noise_std'] = noise
            config['dropout_rate'] = dropout
            config['run_name'] = f'noise_{noise:.1f}_dropout_{dropout:.1f}'
            experiments.append(config)

    # Executar experimentos
    results = []
    for i, config in enumerate(experiments, 1):
        logger.info(f"\nExperimento {i}/{len(experiments)}: {config['run_name']}")
        result = run_advanced_experiment(config, dirs)
        results.append(result)

    # Criar visualização customizada
    create_noise_dropout_heatmap(results, dirs)

    logger.info(f"\n✅ Exemplo 1 concluído! Resultados em: {dirs['base']}")
    return results


def example_2_learning_rate_sweep():
    """Exemplo 2: Varredura fina de learning rates."""

    logger.info("\n" + "="*80)
    logger.info("EXEMPLO 2: VARREDURA DE LEARNING RATES")
    logger.info("="*80 + "\n")

    dirs = setup_experiment_directories(base_dir="outputs_example_2")

    # Learning rates em escala logarítmica
    learning_rates = np.logspace(-4, -2, 10)  # 10 valores entre 1e-4 e 1e-2

    results = []

    for lr in learning_rates:
        config = {
            'run_name': f'lr_{lr:.6f}',
            'initializer': 'he',
            'optimizer': 'adam',
            'learning_rate': float(lr),
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

        logger.info(f"\nTestando LR: {lr:.6f}")
        result = run_advanced_experiment(config, dirs)
        results.append(result)

    # Criar gráfico específico
    create_lr_sweep_plot(results, dirs)

    logger.info(f"\n✅ Exemplo 2 concluído! Resultados em: {dirs['base']}")
    return results


def example_3_activation_comparison():
    """Exemplo 3: Comparação detalhada de ativações."""

    logger.info("\n" + "="*80)
    logger.info("EXEMPLO 3: COMPARAÇÃO DE ATIVAÇÕES")
    logger.info("="*80 + "\n")

    dirs = setup_experiment_directories(base_dir="outputs_example_3")

    # Cada ativação com seu inicializador ideal
    activations_config = [
        ('relu', 'he'),
        ('elu', 'he'),
        ('tanh', 'glorot'),
        ('sigmoid', 'glorot')
    ]

    results = []

    for activation, initializer in activations_config:
        config = {
            'run_name': f'{activation}_{initializer}',
            'initializer': initializer,
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

        logger.info(f"\nTestando: {activation} com {initializer}")
        result = run_advanced_experiment(config, dirs)
        results.append(result)

    # Criar visualização comparativa
    create_activation_comparison_plot(results, dirs)

    logger.info(f"\n✅ Exemplo 3 concluído! Resultados em: {dirs['base']}")
    return results


def example_4_optimal_configuration_search():
    """Exemplo 4: Busca por configuração ótima."""

    logger.info("\n" + "="*80)
    logger.info("EXEMPLO 4: BUSCA DE CONFIGURAÇÃO ÓTIMA")
    logger.info("="*80 + "\n")

    dirs = setup_experiment_directories(base_dir="outputs_example_4")

    # Grid search simplificado
    learning_rates = [5e-4, 1e-3, 2e-3]
    dropouts = [0.0, 0.2, 0.3]
    batch_sizes = [64, 128, 256]

    results = []

    total = len(learning_rates) * len(dropouts) * len(batch_sizes)
    current = 0

    for lr in learning_rates:
        for dropout in dropouts:
            for batch_size in batch_sizes:
                current += 1
                config = {
                    'run_name': f'lr{lr:.0e}_drop{dropout:.1f}_batch{batch_size}',
                    'initializer': 'he',
                    'optimizer': 'adam',
                    'learning_rate': lr,
                    'hidden_units': 64,
                    'batch_size': batch_size,
                    'epochs': 15,
                    'noise_std': 0.2,
                    'dropout_rate': dropout,
                    'activation': 'relu',
                    'seed': 42,
                    'patience': 3,
                    'validation_split': 0.1,
                    'save_model': False
                }

                logger.info(f"\nConfiguracao {current}/{total}: {config['run_name']}")
                result = run_advanced_experiment(config, dirs)
                results.append(result)

    # Encontrar melhor configuração
    best_acc = max(results, key=lambda x: x['clean_accuracy'])
    best_robust = min(results, key=lambda x: x['degradation'])

    logger.info("\n" + "="*80)
    logger.info("MELHORES CONFIGURAÇÕES ENCONTRADAS")
    logger.info("="*80)
    logger.info(f"\n🏆 Melhor Accuracy: {best_acc['run_name']}")
    logger.info(f"   Accuracy: {best_acc['clean_accuracy']:.4f}")
    logger.info(f"   LR: {best_acc['config']['learning_rate']}")
    logger.info(f"   Dropout: {best_acc['config']['dropout_rate']}")
    logger.info(f"   Batch: {best_acc['config']['batch_size']}")

    logger.info(f"\n🛡️  Mais Robusto: {best_robust['run_name']}")
    logger.info(f"   Degradação: {best_robust['degradation']:.2f}%")
    logger.info(f"   LR: {best_robust['config']['learning_rate']}")
    logger.info(f"   Dropout: {best_robust['config']['dropout_rate']}")
    logger.info(f"   Batch: {best_robust['config']['batch_size']}")

    save_results(results, dirs, "grid_search")

    logger.info(f"\n✅ Exemplo 4 concluído! Resultados em: {dirs['base']}")
    return results


# ========== FUNÇÕES DE VISUALIZAÇÃO ==========

def create_noise_dropout_heatmap(results, dirs):
    """Cria heatmap de degradação vs. noise e dropout."""

    # Organizar dados
    noise_levels = sorted(set(r['config']['noise_std'] for r in results))
    dropout_levels = sorted(set(r['config']['dropout_rate'] for r in results))

    # Criar matriz de degradação
    degradation_matrix = np.zeros((len(dropout_levels), len(noise_levels)))

    for result in results:
        noise_idx = noise_levels.index(result['config']['noise_std'])
        dropout_idx = dropout_levels.index(result['config']['dropout_rate'])
        degradation_matrix[dropout_idx, noise_idx] = result['degradation']

    # Plotar heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(degradation_matrix, cmap='RdYlGn_r', aspect='auto')

    # Configurar eixos
    ax.set_xticks(np.arange(len(noise_levels)))
    ax.set_yticks(np.arange(len(dropout_levels)))
    ax.set_xticklabels([f'{n:.1f}' for n in noise_levels])
    ax.set_yticklabels([f'{d:.1f}' for d in dropout_levels])

    ax.set_xlabel('Nível de Ruído (σ)', fontsize=12)
    ax.set_ylabel('Taxa de Dropout', fontsize=12)
    ax.set_title('Degradação (%) vs. Ruído e Dropout', fontsize=14, fontweight='bold')

    # Adicionar valores nas células
    for i in range(len(dropout_levels)):
        for j in range(len(noise_levels)):
            text = ax.text(j, i, f'{degradation_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Degradação (%)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(dirs['plots'] / 'noise_dropout_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_lr_sweep_plot(results, dirs):
    """Cria gráfico detalhado da varredura de learning rate."""

    lrs = [r['config']['learning_rate'] for r in results]
    clean_acc = [r['clean_accuracy'] for r in results]
    noisy_acc = [r['noisy_accuracy'] for r in results]
    degradation = [r['degradation'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Accuracy
    axes[0].semilogx(lrs, clean_acc, 'o-', label='Dados Limpos', linewidth=2)
    axes[0].semilogx(lrs, noisy_acc, 's-', label='Dados Ruidosos', linewidth=2)
    axes[0].set_xlabel('Learning Rate (log scale)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs. Learning Rate', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Degradação
    axes[1].semilogx(lrs, degradation, 'o-', color='red', linewidth=2)
    axes[1].set_xlabel('Learning Rate (log scale)', fontsize=12)
    axes[1].set_ylabel('Degradação (%)', fontsize=12)
    axes[1].set_title('Degradação vs. Learning Rate', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(dirs['plots'] / 'lr_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activation_comparison_plot(results, dirs):
    """Cria gráfico comparativo de ativações."""

    names = [r['run_name'] for r in results]
    clean_acc = [r['clean_accuracy'] for r in results]
    noisy_acc = [r['noisy_accuracy'] for r in results]
    degradation = [r['degradation'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(names))
    width = 0.35

    # Accuracy comparison
    axes[0].bar(x - width/2, clean_acc, width, label='Dados Limpos', alpha=0.8)
    axes[0].bar(x + width/2, noisy_acc, width, label='Dados Ruidosos', alpha=0.8)
    axes[0].set_xlabel('Função de Ativação', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Comparação de Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Degradation
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    axes[1].bar(names, degradation, alpha=0.8, color=colors)
    axes[1].set_xlabel('Função de Ativação', fontsize=12)
    axes[1].set_ylabel('Degradação (%)', fontsize=12)
    axes[1].set_title('Degradação por Ruído', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(dirs['plots'] / 'activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Menu principal para selecionar qual exemplo executar."""

    print("\n" + "="*80)
    print("EXEMPLOS DE SIMULAÇÕES EM BATCH - MLP 64 NEURÔNIOS")
    print("="*80 + "\n")

    print("Exemplos disponíveis:")
    print("1. Comparar ruído com e sem dropout (9 experimentos, ~10-15 min)")
    print("2. Varredura fina de learning rates (10 experimentos, ~15-20 min)")
    print("3. Comparação de funções de ativação (4 experimentos, ~8-12 min)")
    print("4. Busca de configuração ótima (27 experimentos, ~30-40 min)")
    print("5. Executar TODOS os exemplos (~60-90 min)")
    print("0. Sair")

    try:
        choice = input("\nEscolha uma opção (0-5): ").strip()

        if choice == '1':
            example_1_compare_noise_with_dropout()
        elif choice == '2':
            example_2_learning_rate_sweep()
        elif choice == '3':
            example_3_activation_comparison()
        elif choice == '4':
            example_4_optimal_configuration_search()
        elif choice == '5':
            logger.info("\nExecutando TODOS os exemplos...")
            example_1_compare_noise_with_dropout()
            example_2_learning_rate_sweep()
            example_3_activation_comparison()
            example_4_optimal_configuration_search()
            logger.info("\n🎉 Todos os exemplos concluídos!")
        elif choice == '0':
            logger.info("Saindo...")
        else:
            logger.warning("Opção inválida!")

    except KeyboardInterrupt:
        logger.info("\n\nExecução interrompida pelo usuário.")
    except Exception as e:
        logger.error(f"\nErro: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

