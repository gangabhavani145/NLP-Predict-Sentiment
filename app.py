
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# pip install flask_ngrok

# from flask_ngrok import run_with_ngrok

# pip install waitress

# from waitress import serve

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
cv= pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():

  if request.method == 'POST':
		  msg = request.form['message']
		  data = [msg]
		  vect = cv.transform(data).toarray()
		  my_prediction = classifier.predict(vect)
  return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()