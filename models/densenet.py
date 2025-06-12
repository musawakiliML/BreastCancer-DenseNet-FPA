
import numpy as np

from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, 
                                       ReduceLROnPlateau)
from models.fpa_selector import FPAFeatureSelector

class DenseNetFPA:
    """DenseNet-FPA model implementation"""
    
    def __init__(self, num_classes, img_size=(224, 224, 3), 
                 feature_selection_ratio=0.5):
        """
        Initialize DenseNet-FPA model
        
        Args:
            num_classes (int): Number of output classes
            img_size (tuple): Input image dimensions (height, width, channels)
            feature_selection_ratio (float): Ratio of features to select
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.feature_selection_ratio = feature_selection_ratio
        self.model = None
        self.feature_extractor = None
        self.fpa_selector = None
        self.selected_indices = None
        
    def build_feature_extractor(self):
        """Build DenseNet121 feature extractor"""
        base_model = DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.img_size
        )
        
        # Add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Create feature extraction model
        self.feature_extractor = Model(
            inputs=base_model.input, 
            outputs=x
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return self.feature_extractor
    
    def extract_features(self, generator):
        """Extract features using DenseNet"""
        if self.feature_extractor is None:
            self.build_feature_extractor()
            
        features = []
        labels = []
        
        for i in range(len(generator)):
            x, y = generator[i]
            batch_features = self.feature_extractor.predict(x)
            features.append(batch_features)
            labels.append(y)

        # Only concatenate if we have data
        if len(features) > 0:
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
        else:
            raise ValueError("No data was generated - check your data paths and generator configuration")
        
        return features, labels
    
    def select_features(self, X, y):
        """Select features using FPA"""
        n_features = X.shape[1]
        n_select = int(n_features * self.feature_selection_ratio)
        
        self.fpa_selector = FPAFeatureSelector(
            n_features=n_features,
            n_population=20,
            p=0.8,
            alpha=0.1,
            gamma=0.1,
            max_iter=10
        )
        
        best_solution = self.fpa_selector.optimize(X, y)
        self.selected_indices = self.fpa_selector.get_selected_features()
        
        # Ensure we select at least some features
        if len(self.selected_indices) == 0:
            self.selected_indices = np.argsort(best_solution)[-n_select:]
            
        return X[:, self.selected_indices]
    
    def build_model(self):
        """Build the final classification model"""
        if self.feature_extractor is None:
            self.build_feature_extractor()
            
        # Get the output of the feature extractor
        features = self.feature_extractor.output
        
        # Add classification head
        x = Dense(1024, activation='relu')(features)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create final model
        self.model = Model(
            inputs=self.feature_extractor.input,
            outputs=predictions
        )
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, train_generator, val_generator, epochs=50, 
              callbacks=None, learning_rate=1e-4):
        """Train the model"""
        if self.model is None:
            self.build_model(learning_rate)
            
        if callbacks is None:
            callbacks = self.get_default_callbacks()
            
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def train_on_selected_features(self, X_train, y_train, X_val, y_val, epochs=30, learning_rate=1e-4):
        """Train densenet classifier on selected features"""
        input_dim = X_train.shape[1]
        self.classifier = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        self.classifier.compile(
            optimizer=Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = self.classifier.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        return history

    def evaluate_on_selected_features(self, X_test, y_test):
        """Evaluate classifier trained on selected features"""
        y_pred_probs = self.classifier.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = y_test

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        confusion = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion,
            'classification_report': classification_report(y_true, y_pred)
        }
    
    def get_default_callbacks(self):
        """Get default training callbacks"""
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        return [early_stopping, reduce_lr]
    
    def evaluate(self, test_generator):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Get true labels
        y_true = []
        for i in range(len(test_generator)):
            _, y_batch = test_generator[i]
            y_true.append(y_batch)
        y_true = np.concatenate(y_true, axis=0)
        
        # Get predictions
        y_pred = self.model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Classification report
        report = classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=list(test_generator.class_indices.keys())
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics