import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Attention, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, Input, Multiply
import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.utils import class_weight  # For calculating class weights

# 1. Define the emotion classes (ALL CLASSES)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                   'neutral']
num_classes = len(emotion_classes)  # Number of classes (7)

# 2. Define image dimensions (FER-2013 images are 48x48 grayscale)
img_height, img_width = 48, 48

# 3. Define the improved model architecture
def create_model_from_scratch(input_shape=(48, 48, 1), num_classes=7):
    """
    Creates a CNN model for emotion classification.
    This version uses Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Dense, and Attention layers.
    """
    inputs = Input(shape=input_shape, name='input')

    # Convolutional Block 1
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv1_1')(inputs)
    x = BatchNormalization(name='bn1_1')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv1_2')(x)
    x = BatchNormalization(name='bn1_2')(x)
    x = MaxPooling2D((2, 2), name='maxpool1')(x)
    x = Dropout(0.25, name='dropout1')(x)

    # Convolutional Block 2
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv2_1')(x)
    x = BatchNormalization(name='bn2_1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv2_2')(x)
    x = BatchNormalization(name='bn2_2')(x)
    x = MaxPooling2D((2, 2), name='maxpool2')(x)
    x = Dropout(0.25, name='dropout2')(x)

    # Attention Block (applied after Convolutional Block 2)
    x = attention_block(x, 128, block_num=1)  # Name added here

    # Convolutional Block 3
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv3_1')(x)
    x = BatchNormalization(name='bn3_1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv3_2')(x)
    x = BatchNormalization(name='bn3_2')(x)
    x = MaxPooling2D((2, 2), name='maxpool3')(x)
    x = Dropout(0.25, name='dropout3')(x)

    # Convolutional Block 4
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv4_1')(x)
    x = BatchNormalization(name='bn4_1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv4_2')(x)
    x = BatchNormalization(name='bn4_2')(x)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dropout(0.5, name='dropout4')(x)

    # Dense Layers
    x = Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.00005), name='dense1')(x)
    x = BatchNormalization(name='dense_bn')(x)
    x = Dropout(0.5, name='dropout5')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)  # Adjusted num_classes

    model = models.Model(inputs=inputs, outputs=outputs, name='emotion_model')

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.00005),  # Even smaller LR
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

def attention_block(x, filters, block_num):
    """A simple attention block."""
    # Compute attention weights
    attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=f'attention_block_{block_num}_conv')(x)
    x = Multiply(name=f'attention_block_{block_num}_multiply')([x, attention])
    return x

# 4. Define data directories (adjust these paths to your local setup)
train_data_dir = 'data/train/'
val_data_dir = 'data/test/'

# 5. Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    horizontal_flip=True,  # Horizontal flips
    fill_mode='nearest',  # Fill missing pixels
    validation_split=0.2  # 20% for validation
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.2
)  # 20% for validation

def get_class_weights(train_generator):
    """Calculates class weights based on the train generator."""
    class_counts = {}
    for batch_idx in range(train_generator.n // train_generator.batch_size):
        batch = train_generator[batch_idx]
        y_batch = np.argmax(batch[1], axis=1)  # Get class indices
        for class_idx in np.unique(y_batch):
            if class_idx not in class_counts:
                class_counts[class_idx] = 0
            class_counts[class_idx] += np.sum(y_batch == class_idx)

    total_samples = sum(class_counts.values())
    class_weights = {class_idx: total_samples / count
                     for class_idx, count in class_counts.items()}
    
    # Normalize weights
    sum_weights = sum(class_weights.values())
    class_weights = {k: v / sum_weights for k, v in class_weights.items()}
    
    return class_weights

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    color_mode='grayscale',  # Crucial: FER-2013 is grayscale
    class_mode='categorical',  # Multi-class classification
    classes=emotion_classes,  # ALL classes
    subset='training',  # Specify training subset
    shuffle=True  # Shuffle the training data
)

validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    color_mode='grayscale',  # Crucial: FER-2013 is grayscale
    class_mode='categorical',
    classes=emotion_classes,  # ALL classes
    shuffle=False,  # Don't shuffle validation data
    subset='validation'  # Specify validation subset
)

# 6. Create the model
model = create_model_from_scratch(input_shape=(img_height, img_width, 1),
                                 num_classes=num_classes)
model.summary()  # Print model summary to console

# 7. Calculate class weights
# Calculate class weights
class_weights = get_class_weights(train_generator)
print("Class Weights:", class_weights)

# 8. Define training parameters
epochs = 80  # Adjusted epochs
batch_size = 32  # Adjusted batch size

# 9. Calculate steps per epoch and validation steps
train_steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# 10. Callbacks for better training
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20,
                                 restore_best_weights=True, verbose=1)  # Monitor val_accuracy
tensorboard_callback = TensorBoard(log_dir="logs/fit/", histogram_freq=1)

# 11. Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, tensorboard_callback],
    batch_size=batch_size,
    verbose=2,
    class_weight=class_weights  # Add class weights
)

# 12. Create the 'models' directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 13. Save the trained model
model_save_path = os.path.join(models_dir,
                                 'emotion_recognition_model_fer_best.h5')  # Changed filename
model.save(model_save_path)
print(f"Trained model saved to {model_save_path}")

# 14. Create the 'plots' directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 15. Plot training and validation metrics (Separate Plots)
metric_names = ['loss', 'accuracy', 'precision', 'recall']

for idx, metric in enumerate(metric_names):
    plt.figure(figsize=(8, 6))  # Smaller figure size
    plt.plot(history.history[metric], label=f'Training {metric.capitalize()}', linewidth=2)
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}', linewidth=2)
    plt.title(f'Training and Validation {metric.capitalize()}', fontsize=14)  # Smaller title
    plt.xlabel('Epoch', fontsize=12)  # Smaller labels
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xticks(fontsize=10)  # Smaller tick labels
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Save each plot individually
    plot_save_path = os.path.join(plots_dir, f'training_{metric}.png')
    plt.savefig(plot_save_path)
    plt.close()  # Close the figure to free up memory
    print(f"Training plot saved to {plot_save_path}")

# Display class distribution
plt.figure(figsize=(8, 6))
plt.bar(train_generator.class_indices.keys(), np.bincount(train_generator.classes))
plt.title('Training Data Class Distribution', fontsize=14)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plot_save_path = os.path.join(plots_dir, 'class_distribution.png')
plt.savefig(plot_save_path)
plt.show()

print(f"Class distribution plot saved to {plots_save_path}")
