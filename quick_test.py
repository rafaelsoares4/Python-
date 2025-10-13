# quick_test.py
# Script de teste rápido para validar a instalação e executar um experimento mínimo

import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""
    logger.info("Verificando dependências...")

    required_packages = {
        'tensorflow': '2.x',
        'numpy': 'qualquer',
        'matplotlib': 'qualquer',
        'sklearn': 'qualquer'
    }

    missing_packages = []

    for package, version in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            logger.info(f"✅ {package} instalado")
        except ImportError:
            logger.error(f"❌ {package} NÃO encontrado")
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"\nPacotes faltando: {', '.join(missing_packages)}")
        logger.error("Execute: pip install -r requirements.txt")
        return False

    logger.info("✅ Todas as dependências estão instaladas!\n")
    return True


def run_quick_test():
    """Executa um teste rápido com configuração mínima."""
    logger.info("="*80)
    logger.info("EXECUTANDO TESTE RÁPIDO")
    logger.info("="*80 + "\n")

    try:
        from advanced_experiments import (
            setup_experiment_directories,
            run_advanced_experiment
        )

        # Criar diretórios
        dirs = setup_experiment_directories(base_dir="outputs_quick_test")
        logger.info(f"Diretórios criados em: {dirs['base']}\n")

        # Configuração mínima para teste rápido
        config = {
            'run_name': 'quick_test',
            'initializer': 'he',
            'optimizer': 'adam',
            'learning_rate': 1e-3,
            'hidden_units': 64,
            'batch_size': 256,  # Batch maior para ser mais rápido
            'epochs': 3,        # Apenas 3 épocas para teste
            'noise_std': 0.2,
            'dropout_rate': 0.0,
            'activation': 'relu',
            'seed': 42,
            'patience': 2,
            'validation_split': 0.1,
            'save_model': False
        }

        logger.info("Configuração do teste:")
        logger.info(f"  - Épocas: {config['epochs']} (reduzido para teste rápido)")
        logger.info(f"  - Batch size: {config['batch_size']}")
        logger.info(f"  - Hidden units: {config['hidden_units']}")
        logger.info(f"  - Ruído: {config['noise_std']}\n")

        # Executar experimento
        result = run_advanced_experiment(config, dirs)

        # Exibir resultados
        logger.info("\n" + "="*80)
        logger.info("RESULTADOS DO TESTE RÁPIDO")
        logger.info("="*80)
        logger.info(f"✅ Accuracy (dados limpos):   {result['clean_accuracy']:.4f}")
        logger.info(f"✅ Accuracy (dados ruidosos): {result['noisy_accuracy']:.4f}")
        logger.info(f"✅ Degradação:                {result['degradation']:.2f}%")
        logger.info(f"✅ Tempo de treinamento:      {result['training_time']:.2f}s")
        logger.info(f"✅ Épocas treinadas:          {result['epochs_trained']}")
        logger.info("="*80 + "\n")

        logger.info("🎉 TESTE CONCLUÍDO COM SUCESSO!")
        logger.info(f"📁 Resultados salvos em: {dirs['base']}")
        logger.info("\nVocê está pronto para executar os experimentos completos:")
        logger.info("  - python advanced_experiments.py          # Todos os experimentos")
        logger.info("  - python run_custom_experiments.py --help # Ver opções")

        return True

    except Exception as e:
        logger.error(f"\n❌ Erro durante o teste: {str(e)}")
        logger.error("Verifique se todas as dependências estão instaladas corretamente.")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Função principal."""
    logger.info("🧪 Script de Teste Rápido - MLP MNIST\n")

    # Verificar dependências
    if not check_dependencies():
        logger.error("\nPor favor, instale as dependências antes de continuar.")
        sys.exit(1)

    # Executar teste rápido
    if run_quick_test():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

