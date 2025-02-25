from flask import Flask, render_template, request
import pickle
import numpy as np

# Load models and scalers

ms = pickle.load(open("ship_scaler_final.pkl", "rb"))  # MinMaxScaler or StandardScaler
model = pickle.load(open("Shipping_rf_final.pkl", "rb"))  # RandomForest Model


app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("ship1.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    try:
        # Get form data
        Cost_of_the_Product = request.form["Cost_of_the_Product"]
        Discount_offered = request.form["Discount_offered"]
        Prior_purchases = request.form["Prior_purchases"]
        Weight_in_gms = request.form["Weight_in_gms"]
        Product_importance = request.form["Product_importance"]
        Customer_rating = request.form["Customer_rating"]
        Customer_care_calls = request.form["Customer_care_calls"]

        # Encode Product_importance
        importance_mapping = {'low': 0.0, 'medium': 1.0, 'high': 2.0}
        Product_importance_encoded = importance_mapping.get(Product_importance.lower(), 0.0)

        # Prepare input data
        preds = [[
            float(Cost_of_the_Product), 
            float(Customer_rating), 
            int(Customer_care_calls), 
            int(Prior_purchases), 
            Product_importance_encoded, 
            float(Discount_offered), 
            float(Weight_in_gms)
        ]]

        # Transform and predict
        transformed_preds = ms.transform(preds)
        prediction = model.predict(transformed_preds)
        prediction_proba = model.predict_proba(transformed_preds)[0]

        not_reach_prob = prediction_proba[0]
        reach_prob = prediction_proba[1]

        prediction_text = 'There is a {:.2f}% chance that your product will reach in time'.format(reach_prob * 100)
        print(prediction_text)
        print(prediction)

        return render_template("ship1.html", prediction_text=prediction_text)
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
