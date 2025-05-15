# Text-Classification-with-Movie-Reviews 🎬
Text Classification Model for Movie Reviews | Python, TensorFlow, Keras  Engineered a binary sentiment classifier to analyze 50,000 IMDB movie reviews, achieving an 88% predictive accuracy.  Constructed an automated NLP pipeline utilizing TensorFlow Hub pre-trained word embeddings to translate raw text into scalable numerical representations.

Model Architecture
The neural network acts as a "funnel," compressing 50-dimensional word embeddings down to a single binary prediction:

Embedding Layer: google/nnlm-en-dim50/2 (TensorFlow Hub). Maps raw text into 50-dimensional mathematical vectors. Set to trainable=True to fine-tune the embeddings to movie-specific vocabulary.

Hidden Layer: A Dense layer with 16 units and a ReLU activation function to learn non-linear patterns.

Output Layer: A Dense layer with 1 unit (no activation, outputs raw logits) to predict the final binary sentiment.

Dataset
Trained on the IMDB Reviews Dataset via tensorflow_datasets.

Training Set: 15,000 reviews

Validation Set: 10,000 reviews

Test Set: 25,000 reviews

Setup & Installation
Because this project utilizes tensorflow_hub, it requires TensorFlow to run in Keras 2 (Legacy) mode.

Requirements:

tensorflow

tensorflow-hub

tensorflow-datasets

matplotlib

numpy

Bash
pip install tensorflow tensorflow-hub tensorflow-datasets matplotlib numpy
Usage
To use the model to predict your own custom movie reviews, ensure you enable the legacy Keras flag before importing TensorFlow:

Python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub

# Load your saved model (assuming you saved it as 'sentiment_model')
# model = tf.keras.models.load_model('sentiment_model')

custom_reviews = [
    "An absolute joy to watch from start to finish. Breathtaking!",
    "Terrible acting, boring plot, and a complete waste of time."
]

# Convert to tensor and predict
custom_data = tf.constant(custom_reviews)
predictions = model.predict(custom_data)

# Interpret results
for i in range(len(custom_reviews)):
    score = tf.sigmoid(predictions[i][0]).numpy()
    sentiment = "POSITIVE" if score >= 0.5 else "NEGATIVE"
    print(f"Review: {custom_reviews[i]}\nVerdict: {sentiment} ({score*100:.2f}% confidence)\n")
Model Performance
Test Accuracy: ~85.7%

Loss Function: BinaryCrossentropy(from_logits=True)

Optimizer: Adam

Note: The model achieves peak generalization around Epoch 7. Training beyond this point shows signs of overfitting on the training data, as visualized by the divergence of training and validation loss.
