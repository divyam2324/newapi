
import os
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

URGENCY_MAP = {
    'Diabetes': 'Moderate - Schedule a vet visit',
    'Cancer': 'High - Consult a specialist',
    'PBFD': 'High - Isolation required',
    'Egg-Binding': 'Critical - Emergency visit needed'
}

# Load the model
import joblib
model = joblib.load('pet_health_model.pkl')

@app.route('/')
def home():
    return "Pet Health API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get('symptoms', '')

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        prediction = model.predict([symptoms])[0]
        urgency = URGENCY_MAP.get(prediction, 'Medium - Consult a veterinarian')

        return jsonify({
            'symptoms': symptoms,
            'disease': str(prediction),
            'urgency': urgency
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render provides a PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
