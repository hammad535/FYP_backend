from flask import Flask, request, jsonify
from fuzzywuzzy import fuzz
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the test recommender model
test_recommender_model = joblib.load('rfc_model2.joblib')

# Load the anemia model
anemia_model = joblib.load('anemia.joblib')

# Load the diabetes model
diabetes_model = joblib.load('Diabetes.joblib')

# Load the hyperparathyroidism model
hyperparathyroidism_model = joblib.load('Hyperparathyroidism.joblib')


@app.route('/predict/test-recommender', methods=['POST'])
def predict_test_recommender():
    try:
        user_input = request.get_json()
        symptoms_df = pd.read_csv(r'C:\Users\hammad\Desktop\Updated\backend\symptoms_with_ids.csv')

        matched_symptoms = []
        for user_symptom, severity in user_input.items():
            best_match = max(symptoms_df['Symptom'], key=lambda symptom: fuzz.partial_ratio(user_symptom, symptom))
            matched_symptoms.append((best_match, float(severity)))

        symptoms_dict = {}
        for symptom, value in matched_symptoms:
            symptoms_dict[symptom] = value

        symptom_vector = [symptoms_dict[symptom] if symptom in symptoms_dict else 0 for symptom in symptoms_df['Symptom']]
        symptoms_vector = [symptom_vector]

        predicted_test_group_probabilities = test_recommender_model.predict_log_proba(symptoms_vector).tolist()

        response = {
            "predicted_test_group_probabilities": [
                [max(log_prob, -10) for log_prob in probs] for probs in predicted_test_group_probabilities
            ]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/anemia', methods=['POST'])
def predict_anemia():
    try:
        user_input = request.get_json()
        input_array = [float(value) for value in user_input.values()]
        input_array = [input_array]  # Reshape array to match the shape expected by the model

        prediction = anemia_model.predict(input_array).tolist()

        response = {
            "prediction": prediction
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        user_input = request.get_json()
        input_array = [float(value) for value in user_input.values()]
        input_array = [input_array]  # Reshape array to match the shape expected by the model

        prediction = diabetes_model.predict(input_array).tolist()

        response = {
            "prediction": prediction
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/hyperparathyroidism', methods=['POST'])
def predict_hyperparathyroidism():
    try:
        user_input = request.get_json()
        input_array = [float(value) for value in user_input.values()]
        input_array = [input_array]  # Reshape array to match the shape expected by the model

        prediction = hyperparathyroidism_model.predict(input_array).tolist()

        response = {
            "prediction": prediction
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
