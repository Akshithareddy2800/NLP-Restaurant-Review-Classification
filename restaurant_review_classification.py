# Restaurant Review Sentiment Classification
# This script performs binary sentiment analysis (positive/negative) on restaurant reviews using a deep learning model (LSTM) implemented with Keras/TensorFlow.
#
# The workflow includes:
#   - Data loading and exploration
#   - Text preprocessing, tokenization, and sequence padding
#   - Building a neural network with an Embedding layer and LSTM for sequence modeling
#   - Model training, evaluation, and prediction
#
# Usage:
#   python restaurant_review_classification.py
#
# Requirements:
#   - Python 3.x
#   - TensorFlow, pandas, numpy, scikit-learn
#   - 'Restaurant_Reviews.tsv' in the same directory

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Load and Explore the Data
# ----------------------------
# The dataset is a tab-separated file with two columns:
#   - 'Review': The text of the restaurant review
#   - 'Liked': The sentiment label (1 = positive, 0 = negative)
# This is a classic supervised learning problem for binary classification.
df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')
print('First 5 rows of the dataset:')
print(df.head())

# 2. Preprocess the Data
# ----------------------
# Check for missing values to ensure data integrity.
print('\nMissing values in each column:')
print(df.isnull().sum())

# Convert labels to integer type for compatibility with Keras.
df['Liked'] = df['Liked'].astype(int)

# Separate features (X) and labels (y).
X = df['Review'].values  # Array of review texts
y = df['Liked'].values   # Array of binary sentiment labels

# Split the data into training and test sets (80% train, 20% test).
# This helps evaluate the model's generalization to unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Tokenization and Padding
# ---------------------------
# Neural networks require numerical input. We convert text to sequences of integers using a Tokenizer.
# - max_words: Maximum vocabulary size. Words outside this set are replaced with a special <OOV> (out-of-vocabulary) token.
# - max_len: All sequences are padded/truncated to this length for uniformity (required for batch processing in Keras).
max_words = 5000  # Vocabulary size: balances expressiveness and overfitting risk
max_len = 100     # Sequence length: covers most reviews, truncates long ones

# Fit the tokenizer on the training data to build the word index.
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert text reviews to sequences of integer word indices.
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure all inputs are the same length (required for efficient GPU computation).
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# 4. Build the LSTM Model
# -----------------------
# The model architecture is as follows:
#   - Embedding Layer: Learns a dense vector representation for each word (word embeddings), capturing semantic similarity.
#   - LSTM Layer: Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) well-suited for sequence data. It captures long-range dependencies and context in text.
#   - Dropout Layer: Regularization to prevent overfitting by randomly dropping units during training.
#   - Dense Output Layer: Single neuron with sigmoid activation for binary classification (outputs probability of positive sentiment).
#
# Model hyperparameters:
#   - Embedding output_dim: 64 (size of word vectors)
#   - LSTM units: 64 (number of memory cells)
#   - Dropout rate: 0.5 (50% of units dropped during training)
#   - Loss: binary_crossentropy (for binary classification)
#   - Optimizer: Adam (adaptive learning rate)
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),  # Embedding layer: maps word indices to dense vectors
    LSTM(64, return_sequences=False),                                    # LSTM: processes the sequence, outputs a single vector
    Dropout(0.5),                                                       # Dropout: regularization
    Dense(1, activation='sigmoid')                                      # Output: probability of positive sentiment
])

# Compile the model. The metrics include accuracy for easy interpretability.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary to show architecture and parameter counts.
print('\nModel Summary:')
model.summary()

# 5. Train the Model
# ------------------
# Train for 10 epochs (full passes over the training data), with a batch size of 32.
# validation_split=0.2 reserves 20% of the training data for validation (to monitor overfitting).
print('\nTraining the model...')
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# 6. Evaluate the Model
# ---------------------
# Predict probabilities on the test set. Threshold at 0.5 for binary output.
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

# Accuracy: proportion of correct predictions.
# Classification report: includes precision, recall, f1-score for both classes.
print('\nTest Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# 7. Make Predictions on New Reviews
# ----------------------------------
def predict_review(review):
    """
    Predict the sentiment of a single review string using the trained model.
    Steps:
      - Tokenize and pad the input review
      - Predict probability with the model
      - Return 'Positive' if probability > 0.5, else 'Negative'
    """
    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0][0]
    return 'Positive' if pred > 0.5 else 'Negative'

# Example predictions to demonstrate model usage.
print('\nExample predictions:')
print(f"Review: 'The food was amazing and the service was excellent!' => {predict_review('The food was amazing and the service was excellent!')}")
print(f"Review: 'I did not like the food at all.' => {predict_review('I did not like the food at all.')}") 