import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the saved model
model = tf.keras.models.load_model('alexnet_best.keras')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(num_images=5):
    # Load test dataset
    _, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)
    
    # Get some test images
    test_images = []
    test_labels = []
    for image, label in test_ds.take(num_images):
        # Preprocess image
        processed_image = tf.cast(image, tf.float32)
        processed_image = tf.image.resize(processed_image, [227, 227])
        processed_image = processed_image / 255.0
        
        # Make prediction
        prediction = model.predict(tf.expand_dims(processed_image, 0))
        pred_label = np.argmax(prediction)
        
        # Plot
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(f'True: {class_names[label]}')
        plt.axis('off')
        
        # Prediction probabilities
        plt.subplot(1, 2, 2)
        plt.bar(class_names, prediction[0])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Prediction: {class_names[pred_label]}\nConfidence: {prediction[0][pred_label]:.2%}')
        
        plt.tight_layout()
        plt.show()

def create_confusion_matrix():
    # Load test dataset
    _, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)
    
    # Process test dataset
    test_ds = test_ds.map(lambda x, y: (
        tf.cast(tf.image.resize(x, [227, 227]), tf.float32) / 255.0, y
    )).batch(32)
    
    # Get predictions
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plot evaluation metrics
print("Model Evaluation:")
print("-" * 50)
_, test_ds = load_dataset()  # Reuse the data loading function from previous code
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_accuracy:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# Visualize training history
plot_training_history(history)

# Show predictions on sample images
print("\nSample Predictions:")
print("-" * 50)
visualize_predictions(5)

# Create and display confusion matrix
print("\nConfusion Matrix:")
print("-" * 50)
create_confusion_matrix()
