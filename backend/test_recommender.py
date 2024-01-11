from fuzzywuzzy import fuzz
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data from Excel file
data = pd.read_excel(r'C:\Users\hammad\Desktop\Updated\backend\SymptomsWithVaiations.xlsx')

# Shuffle the dataframe
shuffled_dataframe = data.sample(frac=1.0, random_state=42)
shuffled_dataframe.reset_index(drop=True, inplace=True)

# Load the symptom dictionary
symptoms_df = pd.read_csv(r'C:\Users\hammad\Desktop\Updated\backend\symptoms_with_ids.csv')

# Split the data into feature variables (X) and the target variable (y)
X = shuffled_dataframe.drop(columns=['target'])
y = shuffled_dataframe['target']

# Create and train the Random Forest model
rfc = RandomForestClassifier(n_estimators=100, random_state=0)  # You can adjust hyperparameters as needed
rfc.fit(X, y)

# Save the trained RandomForestClassifier model using joblib
model_filename = 'rfc_model2.joblib'
joblib.dump(rfc, model_filename)

# Accept user input for symptoms and their severities
user_input = input("Enter symptoms and their severities separated by commas (e.g., symptom1-0.25,symptom2-0.5,symptom3-1): ")
user_symptoms_severities = [tuple(symptom.split('-')) for symptom in user_input.split(',')]

print('Entered symptoms and severities are: ', user_symptoms_severities)

# Match user input symptoms with the symptom dictionary using fuzzywuzzy
matched_symptoms = []
for user_symptom, severity in user_symptoms_severities:
    best_match = max(symptoms_df['Symptom'], key=lambda symptom: fuzz.partial_ratio(user_symptom, symptom))
    matched_symptoms.append((best_match, float(severity)))

print(matched_symptoms)
symptoms_dict = {}

for symptom, value in matched_symptoms:
    symptoms_dict[symptom] = value

symptom_vector = [symptoms_dict[symptom] if symptom in symptoms_dict else 0 for symptom in symptoms_df['Symptom']]
symptoms_vector = [symptom_vector]

print("vector", symptoms_vector)

# Use the trained RFC model to predict the test group and probabilities
predicted_test_group_rfc = rfc.predict(symptoms_vector)
predicted_test_group_probabilities_rfc = rfc.predict_proba(symptoms_vector)

# Print the predicted test group probabilities for RFC
print(f"Predicted Test Group Probabilities (RFC):")
for i, prob in enumerate(predicted_test_group_probabilities_rfc[0]):
    print(f"Test Group {i+1}: {prob:.2f}")
