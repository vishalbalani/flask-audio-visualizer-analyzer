import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
import pickle as pk



def extract_feature(file_name, **kwargs):
    
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


def gender_p(audio_path):
    from utils import load_data, split_data, create_model
    model = create_model()
    # load the saved/trained weights
    model.load_weights("model.h5")
    # if not file or not os.path.isfile(file):
    #     # if file not provided, or it doesn't exist, use your voice
    #     print("Please talk")
    #     # put the file name here
    file = audio_path
        # record the file (start talking)
        # record_to_file(file)
    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    # show the result!
    finalOut=gender
    # probb_male = male_prob*100
    # probb_female = female_prob*100
    a = (f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
    return finalOut,a