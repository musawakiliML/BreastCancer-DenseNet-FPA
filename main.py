"""
DenseNet-FPA: Integrating DenseNet and Flower Pollination Algorithm for Breast Cancer Histopathology Image Classification

Author: Musa Adamu Wakili
Email:musaadamuw@gmail.com
"""

import matplotlib.pyplot as plt
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

from data.dataloader import DataLoader
from models.densenet import DenseNetFPA

def main():
    
    """Main execution function"""
    
    # Configuration
    config = {
        'data_dir': "/kaggle/input/bach-breast-cancer-histology-images/ ICIAR2018_BACH_Challenge", # Change this to your dataset path
        'img_size': (224, 224),
        'batch_size': 32,
        'feature_selection_ratio': 0.5,
        'learning_rate': 1e-4,
        'epochs': 50
    }

    # Initialize data loader
    data_loader = DataLoader(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size']
    )

    # Load dataset (BACH or BreakHis)
    train_generator, val_generator, test_generator = data_loader.load_bach() #0r data_loader.load_breakhis()

    # Initialize DenseNetFPA model
    model = DenseNetFPA(
        num_classes=data_loader.num_classes,
        img_size=config['img_size'] + (3,),
        feature_selection_ratio=config['feature_selection_ratio']
    )

    # Feature extraction
    print("Extracting features...")
    X_train, y_train = model.extract_features(train_generator)
    X_val, y_val = model.extract_features(val_generator)
    X_test, y_test = model.extract_features(test_generator)

    # Feature selection
    print("Selecting features with FPA...")
    X_train_selected = model.select_features(X_train, y_train)
    X_val_selected = X_val[:, model.selected_indices]
    X_test_selected = X_test[:, model.selected_indices]

    # Train classifier on selected features
    print("Training classifier on selected features...")
    history = model.train_on_selected_features(
        X_train_selected, y_train,
        X_val_selected, y_val,
        epochs=30,  # You can change this
        learning_rate=config['learning_rate']
    )

    # Plot training curves
    plot_training_history(history)

    # Evaluation
    print("Evaluating model on test set...")
    metrics = model.evaluate_on_selected_features(X_test_selected, y_test)

    print("\nEvaluation Results (FPA Features):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("\nClassification Report:")
    print(metrics['classification_report'])


def plot_training_history(history, title='Training and Validation'):
    """Plot training and validation accuracy/loss from Keras history object"""
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'{title} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()