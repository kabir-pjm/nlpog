import joblib
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load trained models
try:
    model_dir = os.path.dirname(__file__)
    salary_model = joblib.load(os.path.join(model_dir, "salary_model.pkl"))
    fraud_model = joblib.load(os.path.join(model_dir, "fraud_model.pkl"))
    ranking_vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    clustering_model = joblib.load(os.path.join(model_dir, "resume_cluster_model.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    # Load PCA model if it exists (for clustering)
    pca_model_path = os.path.join(model_dir, "pca_model.pkl")
    pca_model = joblib.load(pca_model_path) if os.path.exists(pca_model_path) else None
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    print("Please run resume.py first to generate the model files.")
    raise  

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    try:
        # Support both JSON and form data
        if request.is_json:
            data = request.json
        else:
            data = request.form
        
        degree = data.get('degree', 'Unknown')
        experience = float(data.get('experience_years', 0))

        # Encode degree using LabelEncoder
        try:
            degree_encoded = label_encoder.transform([degree])[0]
        except ValueError:
            # If degree not in encoder, use first available
            degree_encoded = 0
        
        prediction = salary_model.predict([[degree_encoded, experience]])[0]
        
        return jsonify({'predicted_salary': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    try:
        # Support both JSON and form data
        if request.is_json:
            data = request.json
        else:
            data = request.form
        
        experience = float(data.get('experience_years', 0))

        # Fraud detection logic
        fraud_prediction = fraud_model.predict([[experience]])[0]
        return jsonify({'fraud_flag': int(fraud_prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/rank', methods=['POST'])
@app.route('/rank_resumes', methods=['POST'])
def rank_resumes():
    try:
        # Support both JSON and form data
        if request.is_json:
            data = request.json
            job_description = data.get('job_description', '')
            resumes = data.get('resumes', [])
            if isinstance(resumes, str):
                resumes = [r.strip() for r in resumes.split(',')]
        else:
            data = request.form
            job_description = data.get('job_description', '')
            resumes = data.getlist('resumes')
            if not resumes:
                resumes = data.get('resumes', '').split(',')

        if not job_description or not resumes:
            return jsonify({'error': 'job_description and resumes are required'}), 400

        job_vector = ranking_vectorizer.transform([job_description])
        resume_vectors = ranking_vectorizer.transform(resumes)
        scores = cosine_similarity(job_vector, resume_vectors).flatten()
        
        ranked_resumes = [{'resume': resume, 'score': float(score)} 
                         for resume, score in sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)]
        
        return jsonify({'ranked_resumes': ranked_resumes})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/cluster_resumes', methods=['POST'])
def cluster_resumes():
    try:
        if pca_model is None:
            return jsonify({'error': 'PCA model not found. Please retrain models using resume.py'}), 500
        
        # Support both JSON and form data
        if request.is_json:
            data = request.json
            resumes = data.get('resumes', [])
            if isinstance(resumes, str):
                resumes = [r.strip() for r in resumes.split(',')]
        else:
            data = request.form
            resumes = data.getlist('resumes')
            if not resumes:
                resumes = data.get('resumes', '').split(',')
        
        if not resumes:
            return jsonify({'error': 'resumes are required'}), 400
        
        resume_vectors = ranking_vectorizer.transform(resumes)
        # Apply PCA reduction before clustering
        reduced_vectors = pca_model.transform(resume_vectors.toarray())
        clusters = clustering_model.predict(reduced_vectors).tolist()
        
        result = [{'resume': resume, 'cluster': int(cluster)} 
                 for resume, cluster in zip(resumes, clusters)]
        
        return jsonify({'resume_clusters': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
