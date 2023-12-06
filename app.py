from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template('index.html')

def Predictor(to_predict_list):
        loaded_model = joblib.load("model.pkl")
        result = loaded_model.predict([to_predict_list])
        return result[0]

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            to_predict_list = request.form.to_dict()
            print(to_predict_list)
            to_predict_list = list(to_predict_list.values())
            result = Predictor(to_predict_list)  
            
            if int(result) == 1:
                prediction = "Retained"
            else:
                prediction = "Churned"
            return render_template("result.html", prediction_text=prediction)
        except Exception as e:
            return render_template("result.html", prediction_text='Error processing the input.')

if __name__ == '__main__':
    app.run(debug=True)