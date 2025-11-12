import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import xgboost as xgb

# Download stopwords
nltk.download('stopwords')

# Load dataset
import os
dataset_path = os.path.join(os.path.dirname(__file__), "cleaned_resume_dataset.csv")
cleaned_df = pd.read_csv(dataset_path)

# Exploratory Data Analysis (EDA)
print(cleaned_df.info())
print(cleaned_df.describe())
print(cleaned_df.isnull().sum())

# Checking column names
print(cleaned_df.columns)

# Rename incorrect column name if needed
if 'experiencere_requirement' in cleaned_df.columns:
    cleaned_df.rename(columns={'experiencere_requirement': 'experience_years'}, inplace=True)

# Extract numeric experience values
cleaned_df['experience_years'] = cleaned_df['experience_years'].astype(str).str.extract(r'(\d+)').astype(float)
cleaned_df['experience_years'].fillna(0, inplace=True)

# Visualizing data distribution
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['experience_years'], bins=30, kde=True)
plt.title("Experience Years Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['matched_score'], bins=30, kde=True)
plt.title("Matched Score Distribution")
plt.show()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
resume_texts = cleaned_df['career_objective'].fillna("").tolist()
resume_vectors = vectorizer.fit_transform(resume_texts)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Resume Ranking & Job Matching
def rank_resumes(job_description, resumes):
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    job_vector = vectorizer.transform([job_description])
    resume_vectors = vectorizer.transform(resumes)
    scores = cosine_similarity(job_vector, resume_vectors).flatten()
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores[ranked_indices]

# Fraudulent Resume Detection
def detect_fraud(df):
    df['fraud_flag'] = df.apply(lambda row: 1 if row.get('experience_years', 0) > 10 and re.search(r'\d{4}', str(row.get('passing_years', ''))) else 0, axis=1)
    return df

cleaned_df = detect_fraud(cleaned_df)

# Resume Clustering
def cluster_resumes(df):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(resume_vectors.toarray())
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)
    df['cluster'] = clusters
    joblib.dump(kmeans, 'resume_cluster_model.pkl')
    joblib.dump(pca, 'pca_model.pkl')  # Save PCA model for inference
    return df

cleaned_df = cluster_resumes(cleaned_df)

# Salary Prediction
label_encoder = LabelEncoder()
cleaned_df['degree_encoded'] = label_encoder.fit_transform(cleaned_df['degree_names'].fillna("Unknown"))
joblib.dump(label_encoder, 'label_encoder.pkl')

def train_salary_model(df):
    features = df[['degree_encoded', 'experience_years']]
    labels = df['matched_score']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Salary Prediction Model Trained!")
    print("Salary Prediction R2 Score:", r2_score(y_test, y_pred))
    print("Salary Prediction Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    joblib.dump(model, 'salary_model.pkl')
    return model

salary_model = train_salary_model(cleaned_df)

# Fraud Detection Model
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df[['experience_years']], cleaned_df['fraud_flag'], test_size=0.3, random_state=42
)
fraud_model = RandomForestClassifier()
fraud_model.fit(X_train, y_train)
y_pred_fraud = fraud_model.predict(X_test)
print("Fraud Detection Accuracy:", accuracy_score(y_test, y_pred_fraud))
joblib.dump(fraud_model, 'fraud_model.pkl')

# Flask Web App
app = Flask(__name__)

@app.route('/rank', methods=['POST'])
def rank():
    data = request.json
    job_desc = data.get('job_description', '')
    resumes = data.get('resumes', [])
    ranked_indices, scores = rank_resumes(job_desc, resumes)
    return jsonify({'ranked_indices': ranked_indices.tolist(), 'scores': scores.tolist()})

@app.route('/detect_fraud', methods=['POST'])
def detect_fraud_api():
    data = request.json
    experience = float(data.get('experience_years', 0))
    fraud_prediction = fraud_model.predict([[experience]])[0]
    return jsonify({'fraud_flag': int(fraud_prediction)})

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    data = request.json
    degree = data.get('degree', 'Unknown')
    experience = float(data.get('experience_years', 0))
    label_encoder = joblib.load('label_encoder.pkl')
    degree_encoded = label_encoder.transform([degree])[0]
    prediction = salary_model.predict([[degree_encoded, experience]])[0]
    return jsonify({'predicted_salary': float(prediction)})

if __name__ == '__main__':
    input_mode = input("Choose mode (flask/manual): ")
    if input_mode.lower() == "manual":
        job_description = input("Enter job description: ")
        resumes = [input("Enter resume text: ") for _ in range(int(input("Number of resumes: ")))]
        ranked_indices, scores = rank_resumes(job_description, resumes)
        print("Ranked Resumes:", ranked_indices.tolist())
        print("Scores:", scores.tolist())
    else:
        app.run(debug=True)

print("All tasks completed!")
