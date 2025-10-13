# utils.py
# Funções auxiliares: pré-processamento, ruído e plotagem

import numpy as np
import matplotlib.pyplot as plt


def preprocess_mnist(x_train, x_test):
    """Normaliza e achata as imagens MNIST.

    Args:
        x_train: Array de imagens de treino
        x_test: Array de imagens de teste

    Returns:
        tuple: (X_train, X_test) normalizados e achatados (float32 entre 0 e 1)
    """
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Achatar as imagens
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    return x_train, x_test


def add_gaussian_noise(x, std=0.2, clip_min=0.0, clip_max=1.0, seed=None):
    """Adiciona ruído Gaussiano às imagens (assumindo imagens já normalizadas em [0,1]).

    Args:
        x: np.array shape (N, D) ou (N, H, W) — o código trata corretamente qualquer um dos dois
        std: desvio padrão do ruído
        clip_min: valor mínimo para clipping
        clip_max: valor máximo para clipping
        seed: semente para reproducibilidade

    Returns:
        np.array: Array com ruído adicionado
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(loc=0.0, scale=std, size=x.shape)
    x_noisy = x + noise
    x_noisy = np.clip(x_noisy, clip_min, clip_max)

    return x_noisy


def plot_history(histories, title=None):
    """Plota loss e accuracy para múltiplos históricos.

    Args:
        histories: dict com chave -> history object do Keras (ou dict com 'history')
        title: título opcional para os gráficos
    """
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    for k, h in histories.items():
        vals = h.history['loss'] if hasattr(h, 'history') else h['loss']
        plt.plot(vals, label=f"{k}")
    plt.title('Loss por época' if title is None else f'Loss - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    for k, h in histories.items():
        vals = h.history['accuracy'] if hasattr(h, 'history') else h['accuracy']
        plt.plot(vals, label=f"{k}")
    plt.title('Accuracy por época' if title is None else f'Accuracy - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_samples(images, labels=None, n_samples=10, title="Amostras"):
    """Plota amostras de imagens MNIST.

    Args:
        images: Array de imagens (N, 784) ou (N, 28, 28)
        labels: Array de labels (opcional)
        n_samples: número de amostras para plotar
        title: título do gráfico
    """
    # Reshape se necessário
    if images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()

    for i in range(min(n_samples, len(images))):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(model, x_test, y_test, verbose=1):
    """Avalia o desempenho do modelo e retorna métricas.

    Args:
        model: modelo treinado
        x_test: dados de teste
        y_test: labels de teste
        verbose: nível de verbosidade

    Returns:
        dict: dicionário com métricas de avaliação
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)

    # Predições
    predictions = model.predict(x_test, verbose=verbose)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Matriz de confusão (simplificada)
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nRelatório de Classificação:")
    print(classification_report(true_classes, predicted_classes))

    print("\nMatriz de Confusão:")
    print(confusion_matrix(true_classes, predicted_classes))

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes
    }