import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    # normalize real temp
    int_features[8] = ((int_features[8] + 8) / 47)
    # normalize feel temp
    int_features[9] = ((int_features[9] + 16) / 66)
    # normalize windspeed
    int_features[-1] = int_features[-1] / 67
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='We predict the number bikes rented to be {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
    
 
    
 
