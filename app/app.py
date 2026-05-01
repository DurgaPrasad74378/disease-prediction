from flask import Flask, render_template, request
import pandas as pd
import sys
import os

# --- BULLETPROOF PATHS ---
# This finds your main "Disease Prediction System" folder automatically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.predict import predict_disease

app = Flask(__name__)

# Safely locate the CSV file using the base directory
csv_path = os.path.join(BASE_DIR, 'data', 'Symptom-severity.csv')
severity_df = pd.read_csv(csv_path)
symptoms_list = severity_df['Symptom'].str.strip().tolist()

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        # 1. Save the exact dropdown choices into a dictionary so we can send them back to the HTML
        user_choices = {
            1: request.form.get('symptom1'),
            2: request.form.get('symptom2'),
            3: request.form.get('symptom3'),
            4: request.form.get('symptom4'),
            5: request.form.get('symptom5')
        }
        
        # 2. Filter out empty selections using our dictionary values
        selected_symptoms = [val for val in user_choices.values() if val != ""]
        
        severity_dict = dict(zip(severity_df['Symptom'].str.strip(), severity_df['weight']))
        symptom_weights = [severity_dict[sym] for sym in selected_symptoms if sym in severity_dict]

        model_file = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
        prediction = predict_disease(symptom_weights, model_path=model_file)

        # 3. Pass 'user_choices' back to the HTML page
        return render_template('index.html', 
                               symptoms=symptoms_list, 
                               prediction_text=f'Most Likely Diagnosis: {prediction}',
                               user_choices=user_choices)

if __name__ == '__main__':
    app.run(debug=True)