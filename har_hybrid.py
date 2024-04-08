import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Permute, Reshape, Multiply, Activation, BatchNormalization
from keras.utils import to_categorical

import tensorflow as tf


class DataPreparation:
    def __init__(self, train_datapath, test_datapath):
        self.df_train = pd.read_csv(train_datapath)
        self.df_test = pd.read_csv(test_datapath)
        self.class_names = None  

    def _encode_categorical(self):
        encoder = LabelEncoder()
        self.df_train['Activity'] = encoder.fit_transform(self.df_train['Activity'])
        self.df_test['Activity'] = encoder.transform(self.df_test['Activity'])
        self.class_names = encoder.classes_ 

    def split_data(self, test_size=0.2, random_state=42):
        self._encode_categorical() 

        features = np.array(self.df_train.iloc[:, 0:-2])
        target = self.df_train.iloc[:, -1]

        # Splitting and expanding dimensions
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=test_size, random_state=random_state)
        X_train, X_val = [np.expand_dims(x, axis=2) for x in [X_train, X_val]]

        # One-Hot Encoding
        y_train, y_val = [to_categorical(y) for y in [y_train, y_val]]

        X_test = np.expand_dims(np.array(self.df_test.iloc[:, 0:-2]), axis=2)
        y_test = to_categorical(self.df_test.iloc[:, -1])

        return X_train, y_train, X_val, y_val, X_test, y_test


# Define Attention layer function
def attention_layer(inputs, time_steps):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# Define Hybrid CNN-LSTM model architecture with an attention mechanism  
class HybridModelTrainer:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_hybrid_model(input_shape, num_classes)

    def build_hybrid_model(self, input_shape, num_classes):

        # CNN layers
        inputs = Input(shape=input_shape)
        cnn_out = Conv1D(filters=64, kernel_size=3)(inputs)
        cnn_out = BatchNormalization()(cnn_out)
        cnn_out = Activation('relu')(cnn_out)
        cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
        cnn_out = Dropout(0.2)(cnn_out)

        cnn_out1 = Conv1D(filters=128, kernel_size=3)(cnn_out)
        cnn_out1 = BatchNormalization()(cnn_out1)
        cnn_out1 = Activation('relu')(cnn_out1)
        cnn_out1 = MaxPooling1D(pool_size=2)(cnn_out1)
        cnn_out1 = Dropout(0.25)(cnn_out1)

        # Apply attention mechanism
        attention_out = attention_layer(cnn_out1, cnn_out1.shape[1])

        # LSTM layers
        lstm_out = LSTM(128, return_sequences=False)(attention_out) 
        lstm_out = Dropout(0.3)(lstm_out)

        # Dense layers for classification/output
        dense_out = Dense(64, activation='relu')(lstm_out)
        output = Dense(num_classes, activation='softmax')(dense_out)

        # Define and compile the model
        model = Model(inputs=[inputs], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Summarize
        model.summary()

        # model.save('./model/hybrid_model.h5') 

        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, patience=5):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[callback])
        
        return history

    def plot_loss_accuracy(self, history):

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy: Training vs. Validation')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss: Training vs. Validation')
        plt.legend()

        # Save the plot to a file
        plt.savefig('./images/loss_accuracy.png')
        plt.close()  
        # plt.show()

    def evaluate_model(self, X_test, y_test, label='Test'):

        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"{label} Accuracy: {accuracy*100:.2f}%")

    def predict_model(self, X_test):
        return self.model.predict(X_test)

    def plot_confusion_matrix(self, y_true, y_pred, classes, label='testing'):
        # Convert one-hot encoded vectors to single integer labels if necessary
        if y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Reds', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix (Percentage) for {label}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # Save the plot to a file
        plt.savefig(f'./images/{label}_confusionmatrix.png')
        plt.close()  
        # plt.show()



train_datapath = 'input/df_train.csv'
test_datapath = 'input/df_test.csv'

data_prep=DataPreparation(train_datapath, test_datapath)

X_train, y_train, X_val, y_val, X_test, y_test = data_prep.split_data(test_size=0.2, random_state=42)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')

print(f'Class names: {data_prep.class_names}')


input_shape = (X_train.shape[1], 1)
num_classes = y_train.shape[1]

model_trainer=HybridModelTrainer(input_shape, num_classes)

# Build model
model= model_trainer.build_hybrid_model(input_shape, num_classes)

# Train and evaluate the model
history = model_trainer.train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, patience=5)

model_trainer.plot_loss_accuracy(history)

model_trainer.evaluate_model(X_val, y_val, 'Validation')

model_trainer.evaluate_model(X_test, y_test, 'Testing')


# Predicting the model
y_val_pred = model_trainer.predict_model(X_val)

y_pred = model_trainer.predict_model(X_test)

# Plotting the confusion matrix
model_trainer.plot_confusion_matrix(y_val, y_val_pred, classes=data_prep.class_names, label='validation')

model_trainer.plot_confusion_matrix(y_test, y_pred, classes=data_prep.class_names, label='testing')


