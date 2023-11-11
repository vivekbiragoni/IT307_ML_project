import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    glucose_level = float(request.form['Glucose Level'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    age = float(request.form['Age'])

    final_features = np.array([glucose_level, insulin, bmi, age])
    
    prediction = model.predict(sc.transform(final_features.reshape(1, -1)))

    if prediction == 1:
       pred = "You have Diabetes, please consult a Doctor."
        
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)