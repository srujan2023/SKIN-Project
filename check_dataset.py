from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define directories
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Adjust size as needed
    batch_size=32,
    class_mode='binary'
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Check some sample images
for data_batch, labels_batch in train_generator:
    # Display first image in the batch
    plt.imshow(data_batch[0])
    plt.title('Sample Image')
    plt.show()
    break  # Display only one batch
