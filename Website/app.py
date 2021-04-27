from flask import Flask, render_template, request,redirect
from pyAudioAnalysis import audioTrainTest as aT
from werkzeug.utils import secure_filename
import pickle
import numpy as np



model = pickle.load(open('LogisticRegression_sentimentAnalysis.pickle','rb'))
audiomodel = pickle.load(open('svmSentimentAnalysis','rb')) 
app = Flask(__name__)

#need to take text and connect to button in html part
@app.route('/')
def main_page():
   return render_template('audio.html',d = "a")

@app.route('/audio')
def audiopage():
   return render_template('audio.html')

@app.route('/text')
def textpage():
   return render_template('home.html')

@app.route("/predictaudio", methods=["GET", "POST"])
def audio():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      d = aT.file_classification(f.filename,"svmSentimentAnalysis","svm")
      print(d)
      per1 = d[1][0]
      per2 = d[1][1]
      if per1 > per2:
         transcript = d[2][0]
      else:
         transcript = d[2][1]
      return render_template('audio.html', d = transcript )
        


@app.route('/predict',methods = ['POST'])
def text():
   data = request.form['text']
   pred = model.predict([data])
   print(pred)
   return render_template('home.html',d = pred[0])


 

if __name__ == '__main__':
   app.run(debug = True)
   