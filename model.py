import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import os
import matplotlib.pyplot as plt
import datetime

# 1. Define the emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                   'neutral']
num_classes = len(emotion_classes)

# 2. Define image dimensions (FER-2013 images are 48x48 grayscale)
img_height, img_width = 48, 48

# 3. Define the model architecture (Improved CNN)
def create_model_from_scratch(input_shape=(48, 48, 1), num_classes=7):
    """
    Creates a more advanced CNN architecture for emotion classification.
    """
    model = models.Sequential([
        # Convolutional Block 1
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional Block 2
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional Block 3
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional Block 4
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        # Dense Layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), # Reduced learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


# 4. Define data directories (adjust these paths to your local setup)
train_data_dir = 'data/train/'
val_data_dir = 'data/test/'

# 5. Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zooms
    horizontal_flip=True,  # Horizontal flips
    fill_mode='nearest',  # Fill missing pixels
    validation_split=0.1  # 10% for validation
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.1
)  # Only rescaling for validation

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    color_mode='grayscale',  # Crucial: FER-2013 is grayscale
    class_mode='categorical',  # Multi-class classification
    classes=emotion_classes,
    subset='training'  # Specify training subset
)

validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    color_mode='grayscale',  # Crucial: FER-2013 is grayscale
    class_mode='categorical',
    classes=emotion_classes,
    shuffle=False,  # Don't shuffle validation data
    subset='validation'  # Specify validation subset
)

# 6. Create the model
model = create_model_from_scratch(input_shape=(img_height, img_width, 1),
                                 num_classes=num_classes)
model.summary()  # Print model summary to console

# 7. Define training parameters
epochs = 80  # Increased epochs
batch_size = 64  # Adjust as needed

# 8. Calculate steps per epoch and validation steps
train_steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# 9. Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, # Increased patience
                                 restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, # Increased patience
                                  min_lr=1e-6, verbose=1)

# 10. TensorBoard callback for logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 11. Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback], # Added TensorBoard
    batch_size=batch_size,
    verbose=2  # Display training progress
)

# 12. Create the 'models' directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 13. Save the trained model
model_save_path = os.path.join(models_dir,
                                 'emotion_recognition_model_fer_v2.h5') # Changed filename
model.save(model_save_path)
print(f"Trained model saved to {model_save_path}")

# 14. Create the 'plots' directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 15. Plot training and validation metrics
plt.figure(figsize=(12, 10))

# Plot training & validation loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation precision
plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Plot training & validation recall
plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# 16. Save the plots to the 'plots' directory
plots_save_path = os.path.join(plots_dir, 'training_metrics_v2.png') # Changed filename
plt.savefig(plots_save_path)
plt.show()

print(f"Training plots saved to {plots_save_path}")