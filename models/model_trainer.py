import pandas as pd
import re
import numpy as np
import nltk
import os
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load training data
train_data_path = "C:/Users/muhil/Downloads/extended_final_data (1).csv"
df = pd.read_csv(train_data_path)

# Preprocess function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
df['Cleaned_Text'] = df['mail'].apply(preprocess_text)

# Convert priority to binary labels
df['Priority_Label'] = df['priority'].apply(lambda x: 1 if str(x).lower() == 'urgent' else 0)

# Print data summary
print(f"Total emails: {len(df)}")
print(f"Priority distribution: {df['Priority_Label'].value_counts()}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
X = vectorizer.fit_transform(df['Cleaned_Text'])
y_priority = df['Priority_Label']

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_model = SVC(kernel='linear', C=1, probability=True)
cv_scores = cross_val_score(svm_model, X, y_priority, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Train final model
X_train, X_test, y_train, y_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
)
svm_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = svm_model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open('models/svm_priority_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")