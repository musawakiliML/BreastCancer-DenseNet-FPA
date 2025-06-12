import tensorflow as tf
import cv2
import glob
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    """Handles loading and preprocessing of BreakHis and BACH datasets"""
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Path to dataset directory
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for data loading
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = None
        self.num_classes = None

    def augment_image(img):
        """Augment images by rotating and mirroring"""
        augmented_images = [img]
        for k in [1, 2]:
            rotated_img = tf.image.rot90(img, k=k)
            mirrored_img = tf.image.flip_left_right(rotated_img)
            augmented_images.append(mirrored_img)
        return augmented_images
    
    def load_and_resize_images(img_list, label, size=(224, 224)):
        """Load images from file paths, resize and augment them"""
        img_array = []
        for img in img_list:
            
            # Read image using OpenCV
            image = cv2.imread(img, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
    
            if label == 0:  # augment benign only
                augmented = self.augment_image(image)
            else:
                augmented = [image]
            
            # Resize and convert to numpy array
            for aug in augmented:
                resized = tf.image.resize(aug, size)
                img_array.append((resized.numpy(), label))
        return img_array
    
    def load_breakhis(base_path='/kaggle/input/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'):
        """Load BreakHis dataset and convert to binary classification (Benign vs Malignant)
        Args:
            base_path (str): Path to the BreakHis dataset directory
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
                   Numpy arrays for training, validation and testing data
        """
        # Get all image paths
        breast_img_paths = glob.glob(f'{base_path}/**/*.png', recursive=True)
    
        # Subtype containers
        A, F, PT, TA = [], [], [], []
        DC, LC, MC, PC = [], [], [], []
        benign, malignant = [], []
    
        # Classify images based on naming convention
        for img in breast_img_paths:
            name = Path(img).name
            if name[6] == 'A':
                A.append(img)
            elif name[6] == 'F':
                F.append(img)
            elif name[6] == 'P' and name[7] == 'T':
                PT.append(img)
            elif name[6] == 'T':
                TA.append(img)
            elif name[6] == 'D':
                DC.append(img)
            elif name[6] == 'L':
                LC.append(img)
            elif name[6] == 'M':
                MC.append(img)
            elif name[6] == 'P':
                PC.append(img)
    
            if name[4] == 'B':
                benign.append(img)
            else:
                malignant.append(img)
    
        # Load and augment
        A_imgs = self.load_and_resize_images(A, 0)
        F_imgs = self.load_and_resize_images(F, 0)
        PT_imgs = self.load_and_resize_images(PT, 0)
        TA_imgs = self.load_and_resize_images(TA, 0)
        DC_imgs = self.load_and_resize_images(DC, 1)
        LC_imgs = self.load_and_resize_images(LC, 1)
        MC_imgs = self.load_and_resize_images(MC, 1)
        PC_imgs = self.load_and_resize_images(PC, 1)
    
        # Combine benign and malignant images
        data_b = A_imgs + F_imgs + PT_imgs + TA_imgs
        data_m = DC_imgs + LC_imgs + MC_imgs + PC_imgs
    
        random.shuffle(data_b)
        random.shuffle(data_m)
    
        # Split benign
        train_b, test_b = train_test_split(data_b, test_size=0.2, random_state=42)
        train_b, val_b = train_test_split(train_b, test_size=0.25, random_state=42)
    
        # Split malignant
        train_m, test_m = train_test_split(data_m, test_size=0.2, random_state=42)
        train_m, val_m = train_test_split(train_m, test_size=0.25, random_state=42)
    
        # Combine all splits
        train_data = train_b + train_m
        val_data = val_b + val_m
        test_data = test_b + test_m
    
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
    
        # Convert to arrays
        X_train, y_train = zip(*train_data)
        X_val, y_val = zip(*val_data)
        X_test, y_test = zip(*test_data)
    
        return (
            np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test)
        )
    
    def load_bach(self):
        """Load BACH dataset and convert to binary classification (Benign vs Malignant)
        
        Returns:
            tuple: (train_generator, val_generator, test_generator) 
                   TensorFlow Dataset objects for training, validation and testing
        """
        # Path configuration
        dataset_path = os.path.join(self.data_dir, "ICIAR2018_BACH_Challenge/Photos")
        self.data_dir = dataset_path
        
        # Load and prepare dataframe with binary classes
        image_paths = []
        labels = []
        for class_name in ["Benign", "InSitu", "Invasive", "Normal"]:
            class_path = os.path.join(dataset_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith(".tif"):
                    image_paths.append(os.path.join(class_name, img_name))
                    # Convert to binary classes:
                    # Benign (including Normal) = 0, Malignant (InSitu+Invasive) = 1
                    if class_name in ["InSitu", "Invasive"]:
                        labels.append("Malignant")
                    else:
                        labels.append("Benign")
        
        # Create dataframe
        image_df = pd.DataFrame({"filename": image_paths, "label": labels})
        
        # Shuffle and split dataset
        image_df = image_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df, temp_df = train_test_split(
            image_df, 
            test_size=0.5, 
            stratify=image_df["label"], 
            random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df["label"], 
            random_state=42
        )
        
        # Create data generators
        train_generator = self.create_generators(train_df, 'training')
        val_generator = self.create_generators(val_df, 'validation')
        
        # For test generator
        test_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image
        )
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=dataset_path,
            x_col='filename',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',  # Changed to binary for 2-class problem
            shuffle=False,
            classes=["Benign", "Malignant"]
        )
        
        # Store the class indices for reference
        self.classes = test_generator.class_indices
        self.num_classes = 1  # Binary classification
        
        return train_generator, val_generator, test_generator
    
    def preprocess_image(self, image):
        """Preprocess single image (resize, normalize, etc.)"""
        image = tf.image.resize(image, self.img_size)
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image
    
    def create_generators(self, df, subset):
        """Create TensorFlow data generators"""
        datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=self.data_dir,
            x_col='filename',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset=subset,
            shuffle=True,
            classes=["Benign", "Malignant"]
        )
        
        self.classes = generator.class_indices
        self.num_classes = len(self.classes)
        
        return generator