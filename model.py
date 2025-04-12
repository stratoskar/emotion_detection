# Import necessary libraries
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

# Define the emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral']
num_classes = len(emotion_classes)  # Number of classes (7)

# Define image dimensions (FER-2013 images are 48x48 grayscale)
img_height, img_width = 48, 48

# Define the model architecture
def create_model_from_scratch(input_shape=(48, 48, 1), num_classes=7):
    """
    Creates a CNN model for emotion classification.
    This version uses Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Dense, and Attention layers.
    """
    # Input layer definition
    inputs = Input(shape=input_shape, name='input')

    # Convolutional Block 1
    # First convolutional layer with 64 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv1_1')(inputs)
    # Batch normalization layer
    x = BatchNormalization(name='bn1_1')(x)
    # Second convolutional layer with 64 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv1_2')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn1_2')(x)
    # Max pooling layer with a 2x2 pool size
    x = MaxPooling2D((2, 2), name='maxpool1')(x)
    # Dropout layer with a 25% dropout rate
    x = Dropout(0.25, name='dropout1')(x)

    # Convolutional Block 2
    # First convolutional layer with 128 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv2_1')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn2_1')(x)
    # Second convolutional layer with 128 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv2_2')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn2_2')(x)
    # Max pooling layer with a 2x2 pool size
    x = MaxPooling2D((2, 2), name='maxpool2')(x)
    # Dropout layer with a 25% dropout rate
    x = Dropout(0.25, name='dropout2')(x)

    # Attention Block (applied after Convolutional Block 2)
    # Apply an attention mechanism to the output of the second convolutional block
    x = attention_block(x, 128, block_num=1)

    # Convolutional Block 3
    # First convolutional layer with 256 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv3_1')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn3_1')(x)
    # Second convolutional layer with 256 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv3_2')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn3_2')(x)
    # Max pooling layer with a 2x2 pool size
    x = MaxPooling2D((2, 2), name='maxpool3')(x)
    # Dropout layer with a 25% dropout rate
    x = Dropout(0.25, name='dropout3')(x)

    # Convolutional Block 4
    # First convolutional layer with 512 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv4_1')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn4_1')(x)
    # Second convolutional layer with 512 filters, 3x3 kernel, same padding, ReLU activation, and L2 regularization
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.00005), name='conv4_2')(x)
    # Batch normalization layer
    x = BatchNormalization(name='bn4_2')(x)
    # Global average pooling layer to reduce spatial dimensions to a single vector
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    # Dropout layer with a 50% dropout rate
    x = Dropout(0.5, name='dropout4')(x)

    # Dense Layers
    # Fully connected layer with 512 units, ReLU activation, and L2 regularization
    x = Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.00005), name='dense1')(x)
    # Batch normalization layer
    x = BatchNormalization(name='dense_bn')(x)
    # Dropout layer with a 50% dropout rate
    x = Dropout(0.5, name='dropout5')(x)
    # Output layer with number of classes units and softmax activation for probability distribution
    outputs = Dense(num_classes, activation='softmax', name='output')(x)  # Adjusted num_classes

    # Create the Keras model with input and output tensors
    model = models.Model(inputs=inputs, outputs=outputs, name='emotion_model')

    # Compile the model with Adam optimizer (learning rate 0.00005), categorical cross-entropy loss, and accuracy, precision, and recall metrics
    model.compile(optimizer=optimizers.Adam(learning_rate=0.00005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

def attention_block(x, filters, block_num):
    """A simple attention block."""
    # Compute attention weights using a 1x1 convolutional layer with sigmoid activation
    attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=f'attention_block_{block_num}_conv')(x)
    # Multiply the input tensor by the attention weights
    x = Multiply(name=f'attention_block_{block_num}_multiply')([x, attention])
    return x

# Define data directories 
# These directories may vary based on your dataset structure
train_data_dir = 'data/train/' 
val_data_dir = 'data/test/'

# Create data generators for training and validation
# ImageDataGenerator for training data with rescaling, horizontal flip, fill mode, and validation split
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values to the range [0, 1]
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Strategy for filling newly created pixels after transformations
    validation_split=0.2  # Reserve 20% of the training data for validation
)

# ImageDataGenerator for validation data with only rescaling and validation split
val_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.2
)  # 20% for validation

def get_class_weights(train_generator):
    """Calculates class weights based on the train generator."""
    # Initialize a dictionary to store the count of each class
    class_counts = {}
    # Iterate through the batches in the training generator
    for batch_idx in range(train_generator.n // train_generator.batch_size):
        # Get a batch of data and labels
        batch = train_generator[batch_idx]
        # Extract the one-hot encoded labels and find the class indices
        y_batch = np.argmax(batch[1], axis=1)  # Get class indices
        # Count the occurrences of each unique class in the current batch
        for class_idx in np.unique(y_batch):
            if class_idx not in class_counts:
                class_counts[class_idx] = 0
            class_counts[class_idx] += np.sum(y_batch == class_idx)

    # Calculate the total number of samples
    total_samples = sum(class_counts.values())
    # Calculate the class weights as the inverse proportion of each class frequency
    class_weights = {class_idx: total_samples / count
                     for class_idx, count in class_counts.items()}
    
    # Normalize weights so they sum to 1 (optional but can be helpful for stability)
    sum_weights = sum(class_weights.values())
    class_weights = {k: v / sum_weights for k, v in class_weights.items()}
    
    return class_weights

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),  # Resize images to the defined dimensions
    batch_size=32,  # Set the batch size
    color_mode='grayscale',  # Specify that the images are grayscale
    class_mode='categorical',  # Encode labels as one-hot vectors
    classes=emotion_classes,  # Specify the order of the emotion classes
    subset='training',  # Indicate that this generator is for the training subset
    shuffle=True  # Shuffle the training data to improve training
)

# Create the validation data generator
validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),  # Resize images to the defined dimensions
    batch_size=32,  # Set the batch size
    color_mode='grayscale',  # Specify that the images are grayscale
    class_mode='categorical',  # Encode labels as one-hot vectors
    classes=emotion_classes,  # Specify the order of the emotion classes
    shuffle=False,  # Do not shuffle the validation data to ensure consistent evaluation
    subset='validation'  # Indicate that this generator is for the validation subset
)

# Create the model
# Instantiate the CNN model defined in the create_model_from_scratch function
model = create_model_from_scratch(input_shape=(img_height, img_width, 1),
                                 num_classes=num_classes)
# Print a summary of the model architecture
model.summary()  # Print model summary to console

# Calculate class weights
# Calculate class weights based on the training data distribution
class_weights = get_class_weights(train_generator)
# Print the calculated class weights
print("Class Weights:", class_weights)

# Define training parameters
epochs = 80  # Number of training epochs
batch_size = 32  # Batch size used during training

# Calculate steps per epoch and validation steps
# Calculate the number of training steps per epoch
train_steps_per_epoch = train_generator.samples // batch_size
# Calculate the number of validation steps
validation_steps = validation_generator.samples // batch_size

# Callbacks for better training
# Early stopping callback to prevent overfitting by monitoring validation accuracy
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20,
                                 restore_best_weights=True, verbose=1)  # Monitor val_accuracy
# TensorBoard callback to log training metrics for visualization
tensorboard_callback = TensorBoard(log_dir="logs/fit/", histogram_freq=1)

# Train the model
# Train the model using the training data generator and validate with the validation data generator
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, tensorboard_callback],  # Use the defined callbacks
    batch_size=batch_size,
    verbose=2,  # Set verbosity level to 2 for detailed output
    class_weight=class_weights  # Apply the calculated class weights to handle class imbalance
)

# Create the 'model' directory if it doesn't exist
models_dir = 'model'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the trained model
# Define the path to save the trained model
model_save_path = os.path.join(models_dir,
                                 'emotion_recognition_model_fer_best.h5') 
# Save the trained model to the specified path
model.save(model_save_path)
# Print a confirmation message
print(f"Trained model saved to {model_save_path}")

# Create the 'plots' directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plot training and validation metrics (Separate Plots)
# Define the list of metrics to plot
metric_names = ['loss', 'accuracy', 'precision', 'recall']

# Iterate through each metric and create a separate plot
for idx, metric in enumerate(metric_names):
    # Create a new figure for each metric
    plt.figure(figsize=(8, 6)) 
    # Plot the training metric
    plt.plot(history.history[metric], label=f'Training {metric.capitalize()}', linewidth=2)
    # Plot the validation metric
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}', linewidth=2)
    # Set the title of the plot
    plt.title(f'Training and Validation {metric.capitalize()}', fontsize=14) 
    # Set the label for the x-axis
    plt.xlabel('Epoch', fontsize=12)
    # Set the label for the y-axis
    plt.ylabel(metric.capitalize(), fontsize=12)
    # Set the font size for x-axis ticks
    plt.xticks(fontsize=10)
    # Set the font size for y-axis ticks
    plt.yticks(fontsize=10)
    # Display the legend
    plt.legend(fontsize=10)
    # Add a grid to the plot
    plt.grid(True)
    # Adjust plot to fit within the figure area
    plt.tight_layout()
    
    # Define the path to save the current plot
    plot_save_path = os.path.join(plots_dir, f'training_{metric}.png')
    # Save the current plot
    plt.savefig(plot_save_path)
    # Close the current figure to free up memory
    plt.close()  # Close the figure to free up memory
    # Print a confirmation message
    print(f"Training plot saved to {plot_save_path}")

# Display class distribution in the training data
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