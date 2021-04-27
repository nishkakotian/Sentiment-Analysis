import argparse
import os
import cv2


from flask import Flask, render_template, request,redirect
from pyAudioAnalysis import audioTrainTest as aT
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
vid = keras.models.load_model('model.h5')
model_json = vid.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model = pickle.load(open('LogisticRegression_sentimentAnalysis.pickle','rb'))
audiomodel = pickle.load(open('svmSentimentAnalysis','rb')) 
app = Flask(__name__)

img_rows, img_cols, frames = 112,112,5
channel = 3 

nb_classes = 2

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=True, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(self.depth)]
        framearray = []

        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            frame = cv2.resize(frame, (self.height, self.width))
            if color:
                framearray.append(frame)
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()
        return np.array(framearray)

vid3d = Videoto3D(img_rows, img_cols, frames)

def loaddata(video_dir, vid3d, nclass, color=False, skip=True):
    #classes = os.listdir(video_dir)
    print(video_dir)
    X = []
    labels = []

    X.append(vid3d.video3d(video_dir, color=color, skip=skip))
    
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1))
    else:
        return np.array(X).transpose((0, 2, 3, 1))

@app.route('/')
def main_page():
   return render_template('home.html',d ="abc")

@app.route('/audio')
def audiopage():
   return render_template('audio.html',d = "abc")

@app.route('/video')
def videopage():
   return render_template('video.html',d="abc")

@app.route('/text')
def textpage():
   return render_template('home.html',d="abc")

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
        

@app.route('/predictvid',methods = ["GET","POST"])
def video():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      x = loaddata(f.filename, vid3d, nb_classes, True, True)
      print(x.shape)
      x=x.reshape((1,112,112,5,3))
      # x = x.reshape(-1)
      print(x.shape)
      # print(x)
      pred= vid.predict(x)
      transcript = "abc"
      if(pred[0][0]>pred[0][1]):
         transcript = "NEGATIVE"
      else :
         transcript = "POSITIVE"
      print(pred)
      return render_template('video.html', d = transcript )

@app.route('/predict',methods = ['POST'])
def text():
   data = request.form['text']
   pred = model.predict([data])
   print(pred)
   return render_template('home.html',d = pred[0])


 
if __name__ == '__main__':
   app.run(debug = True)
   