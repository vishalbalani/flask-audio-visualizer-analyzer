from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pickle as pk
import numpy as np
import os
from sys import byteorder
from array import array
from struct import pack
from gender_pro import *
from graph import *
from compare import *

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER



@app.route("/", methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():
    audiofile = request.files["audioFile"]
    audio_path = "./" + audiofile.filename
    audiofile.save(audio_path)
    
    audiofile1 = request.files["audioFile1"]
    audio_path1 = "./" + audiofile1.filename
    audiofile1.save(audio_path1)
    
    p1=audiofile.filename
    p2=audiofile1.filename
    
    visualize(audio_path,audio_path1)
    finalOut,a=gender_p(audio_path)
    finalOut1,a1=gender_p(audio_path1)
    
    SOURCE_FILE = audio_path
    TARGET_FILE = audio_path1
    b=correlate(SOURCE_FILE, TARGET_FILE)
  
    # path = "freq1.png"
    # freq2 = "freq2.png"
    # spec1 = "spec1.png"
    # spec2 = "spec2.png"
      
    
    
    ff1 = os.path.join(app.config['UPLOAD_FOLDER'], 'freq1.png')
    ff2 = os.path.join(app.config['UPLOAD_FOLDER'], 'freq2.png')
    ss1 = os.path.join(app.config['UPLOAD_FOLDER'], 'spec1.png')
    ss2 = os.path.join(app.config['UPLOAD_FOLDER'], 'spec2.png')
    oo1 = os.path.join(app.config['UPLOAD_FOLDER'], 'oboe1.png')
    oo2 = os.path.join(app.config['UPLOAD_FOLDER'], 'oboe2.png')
    cc1 = os.path.join(app.config['UPLOAD_FOLDER'], 'calrinet1.png')
    cc2 = os.path.join(app.config['UPLOAD_FOLDER'], 'calrinet2.png')
    aa1 = os.path.join(app.config['UPLOAD_FOLDER'], 'first.png')
    aa2 = os.path.join(app.config['UPLOAD_FOLDER'], 'first2.png')
    
    print(ff1)
    return render_template("index.html",b=b, p1=p1, p2=p2, finalOut=finalOut, finalOut1=finalOut1, a=a,  a1=a1, ff1=ff1, ff2=ff2, ss1=ss1, ss2=ss2 , oo1=oo1, oo2=oo2, cc1=cc1, cc2=cc2, aa1=aa1, aa2=aa2)


if __name__ == "__main__":
    app.run(port=5000,debug =True)
    
    