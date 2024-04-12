from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('rf_Model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

import pickle

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)


@app.route('/')
def home():
    car_model_values = df['Car Model'].unique()
    car_name_values = df['Car Name'].unique()
    fuel_type_values = df['Fuel Type'].unique()
    owner_values = df['Owner'].unique()

    return render_template('index.html', 
    car_model_values=car_model_values,
    car_name_values=car_name_values,
    fuel_type_values=fuel_type_values,
    owner_values=owner_values)


@app.route('/predict', methods=['POST'])
def predict():
    Car_name = request.form['car_name']
    Car_model = request.form['car_model']
    Mileage = int(request.form['mileage'])
    Fuel_type = request.form['fuel_type']
    Year = int(request.form['year'])
    Owner = request.form['owner']
    
    input_data = pd.DataFrame({
        'Car Name': [Car_name],
        'Car Model': [Car_model],
        'Year': [Year],
        'Mileage': [Mileage],
        'Fuel Type': [Fuel_type],
        'Owner': [Owner]
    })
    print("Training Data Columns:", df.columns)
    print("Prediction Data Columns:", input_data.columns)
    columns_to_encode = ['Car Name', 'Car Model', 'Owner', 'Fuel Type']
    for col, le in label_encoders.items():
        print(f"Column: {col}, LabelEncoder: {le}")

    for col in columns_to_encode:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Make a prediction using the model
    predicted_price = model.predict(input_data)

    return render_template('result.html', price=predicted_price[0])
if __name__ == '__main__':
    app.run(debug=True)
