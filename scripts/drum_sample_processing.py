import os
import numpy as np
from PIL import Image

# Process the data or simply load processed data
def load_images(folder_path, num_images=None, skip=False):
    images = []
    
    # If you've already processed the data
    if skip:
        
        filenames = os.listdir(folder_path)
        
        for filename in filenames:
            
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)

            # Just load the processed data
            images = np.load(file_path)

            print(type(images))            
            print(images.shape)

    # If you want to process the data     
    else:
        
        # Find all the .png files
        filenames = os.listdir(folder_path)[:num_images] if num_images else os.listdir(folder_path)
        for filename in filenames:
            if filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)

                # Open the images
                img = Image.open(img_path)
                
                # downsample the images
                img_resized = img.resize((100, 100))
                
                # Convert the images to numpy arrays
                imag = np.array(img_resized)

                # Put the dimensions in the correct order
                img_permuted = np.transpose(imag, (2, 0, 1))

                # Cut off the un-needed channels
                img_mids = img_permuted[1:-1, :, :]

                images.append(img_mids) 
                print(f"Successfully reshaped {img_path} into {img_mids.shape}")

        # Save the data 
        output_path = r'.\sample_data\max_drm'
        np.save(os.path.join(output_path, f'data.npy'), images)
            
    return np.array(images)

# Make the label vector
def create_label_vector(num_samples):
    labels = np.tile([0, 1, 2, 3], int(np.ceil(num_samples / 4)))[:num_samples]
    return labels

# Divide the data into training, development, and testing sets
def split_data(images, labels, train_size, dev_size, test_size):
    total_size = train_size + dev_size + test_size
    assert total_size <= len(images), "Total size exceeds the number of available images."

    X_train = images[:train_size]
    y_train = labels[:train_size]

    X_dev = images[train_size:train_size + dev_size]
    y_dev = labels[train_size:train_size + dev_size]

    X_test = images[train_size + dev_size:total_size]
    y_test = labels[train_size + dev_size:total_size]

    return X_train, y_train, X_dev, y_dev, X_test, y_test

# Do all of the above functions
def divvy(folder_path, train_size, dev_size, test_size, skip=False):
    
    total_num = train_size + dev_size + test_size
    
    # Load images
    images = load_images(folder_path, num_images=total_num, skip=skip)

    # Create label vector
    labels = create_label_vector(len(images))

    # Split data
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(
        images, labels, train_size, dev_size, test_size
    )

    # Print out shapes
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_dev shape:", X_dev.shape)
    print("y_dev shape:", y_dev.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

# This code on its own can be used to process the images
if __name__ == "__main__":
    
    skip_process = False
    
    folder_path = r'.\sample_data\total_drm'
        
    train_size = 3200  # Change this value according to your needs
    dev_size = 320   # Change this value according to your needs
    test_size = 480   # Change this value according to your needs

    X_train, y_train, X_dev, y_dev, X_test, y_test = divvy(
        folder_path, train_size, dev_size, test_size, skip=skip_process
    )
