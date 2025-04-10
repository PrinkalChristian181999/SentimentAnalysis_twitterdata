# # modelTrain.py
# from mongoConnect import connectMongoDB
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from cleaningPipeline import cleanText
# import joblib

# def load_training_data():
#     """Load data from MongoDB's sentimentAnalysis collection"""
#     db = connectMongoDB()
#     collection = db["sentimentAnalysis"]
#     data = list(collection.find({}, {"cleanedDescription": 1, "sentimentAnalysis.label": 1}))
    
#     texts = [doc["cleanedDescription"] for doc in data]
#     labels = [doc["sentimentAnalysis"]["label"] for doc in data]
    
#     return texts, labels

# def train_model():
#     # Load data
#     texts, labels = load_training_data()
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         texts, labels, test_size=0.2, random_state=42
#     )
    
#     # Vectorize text
#     vectorizer = TfidfVectorizer(max_features=1000)
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)
    
#     # Train classifier
#     clf = RandomForestClassifier(n_estimators=100)
#     clf.fit(X_train_vec, y_train)
    
#     # Evaluate
#     y_pred = clf.predict(X_test_vec)
#     print(classification_report(y_test, y_pred))
    
#     # Save model and vectorizer
#     joblib.dump(clf, "crime_sentiment_model.pkl")
#     joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
#     print("Model and vectorizer saved successfully")

# def processCrimeUpdateToTrain(analysis):
#     """Optional: Add new data points incrementally to retrain model"""
#     # This can be implemented later for continuous learning
#     pass

# if __name__ == "__main__":
#     train_model()
# modelTrain.py
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from mongoConnect import connectMongoDB
from cleaningPipeline import cleanText

# Load data from MongoDB
def load_training_data():
    db = connectMongoDB()
    collection = db["sentimentAnalysis"]
    data = list(collection.find({}, {"cleanedDescription": 1, "sentimentAnalysis.label": 1}))
    
    texts = [doc["cleanedDescription"] for doc in data]
    labels = [0 if doc["sentimentAnalysis"]["label"] == "Negative" 
              else 1 if doc["sentimentAnalysis"]["label"] == "Neutral" 
              else 2 for doc in data]  # Map to 0=Negative, 1=Neutral, 2=Positive
    
    return texts, np.array(labels)

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def train_bert():
    # Load data
    texts, labels = load_training_data()
    
    # Split data (stratified due to imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Tokenize text
    train_encodings = tokenizer(
        X_train, 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='tf'
    )
    
    test_encodings = tokenizer(
        X_test, 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='tf'
    )
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).shuffle(1000).batch(16)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    )).batch(16)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Train with class weights (to handle imbalance)
    class_counts = np.bincount(y_train)
    class_weights = {i: 1.0/class_counts[i] for i in range(3)}
    
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=3,
        class_weight=class_weights
    )
    
    # Save model
    model.save_pretrained("bert_crime_sentiment")
    tokenizer.save_pretrained("bert_crime_sentiment")
    print("BERT model saved!")

def processCrimeUpdateToTrain(analysis):
    """Use BERT for predictions on new data"""
    # Load saved model
    loaded_model = TFBertForSequenceClassification.from_pretrained("bert_crime_sentiment")
    
    # Preprocess text
    text = analysis["cleanedDescription"]
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)
    
    # Predict
    outputs = loaded_model(inputs)
    prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
    
    print(f"Predicted sentiment: {['Negative', 'Neutral', 'Positive'][prediction]}")

if __name__ == "__main__":
    train_bert()