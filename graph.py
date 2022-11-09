import numpy as np
import matplotlib.pyplot as plot
from scipy import pi
from glob import glob
from scipy.fftpack import fft
import librosa as lr
import librosa.display
from IPython.display import Audio
import IPython.display as ipd
import scipy



def freqdomain(audio1,audio2):
    y, sr = lr.load(audio1)
    signalAmplitude   = np.sin(y)
    plot.subplot(211)
    plot.plot(y, signalAmplitude,'bs')
    plot.xlabel('time')
    plot.ylabel('amplitude')
    plot.subplot(212)
    plot.magnitude_spectrum(signalAmplitude,Fs=4)
    plot.savefig('static/people_photo/freq1.png')

    y, sr = lr.load(audio2)
    signalAmplitude   = np.sin(y)
    plot.subplot(211)
    plot.plot(y, signalAmplitude,'bs')
    plot.xlabel('time')
    plot.ylabel('amplitude')
    plot.subplot(212)
    plot.magnitude_spectrum(signalAmplitude,Fs=4)
    plot.savefig('static/people_photo/freq2.png')

def oboe(audio1,audio2):
    y, sr = lr.load(audio1)
    f = np.linspace(0, sr, 4096)
    print(y.shape)
    X = np.fft.fft(y[10000:14096])
    X_mag = np.absolute(X)
    plot.figure(figsize=(14, 5))
    plot.plot(f[:2000], X_mag[:2000]) # magnitude spectrum
    plot.xlabel('Frequency (Hz)')
    plot.savefig('static/people_photo/oboe1.png')
    

    y, sr = lr.load(audio2)
    f = np.linspace(0, sr, 4096)
    print(y.shape)
    X = np.fft.fft(y[10000:14096])
    X_mag = np.absolute(X)
    plot.figure(figsize=(14, 5))
    plot.plot(f[:2000], X_mag[:2000]) # magnitude spectrum
    plot.xlabel('Frequency (Hz)')
    plot.savefig('static/people_photo/oboe2.png')

def spec(audio1,audio2):

    y, sr = lr.load(audio1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plot.subplot(4, 2, 2)
    lr.display.specshow(D, y_axis='log')
    plot.colorbar(format='%+2.0f dB')
    plot.title('Log-frequency power spectrogram')
    plot.savefig('static/people_photo/spec1.png')

    y, sr = lr.load(audio2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plot.subplot(4, 2, 2)
    lr.display.specshow(D, y_axis='log')
    plot.colorbar(format='%+2.0f dB')
    plot.title('Log-frequency power spectrogram')
    plot.savefig('static/people_photo/spec2.png')
    
def clarinet(audio1,audio2):
    y, sr = lr.load(audio1)
    f = np.linspace(0, sr, 4096)
    ipd.Audio(y, rate=sr)
    print(y.shape)
    X = np.fft.fft(y[10000:14096])
    X_mag = np.absolute(X)
    plot.figure(figsize=(14, 5))
    plot.plot(f[:500], X_mag[:500]) # magnitude spectrum
    plot.xlabel('Frequency (Hz)')
    plot.savefig('static/people_photo/calrinet1.png')
    
    y, sr = lr.load(audio2)
    f = np.linspace(0, sr, 4096)
    ipd.Audio(y, rate=sr)
    print(y.shape)
    X = np.fft.fft(y[10000:14096])
    X_mag = np.absolute(X)
    plot.figure(figsize=(14, 5))
    plot.plot(f[:500], X_mag[:500]) # magnitude spectrum
    plot.xlabel('Frequency (Hz)')
    plot.savefig('static/people_photo/calrinet2.png')
    
def first(audio1,audio2):
    
    


    y, sr = lr.load(audio1)
    ipd.Audio(y, rate=sr)
    plot.figure(figsize=(15, 5))
    lr.display.waveshow(y, sr, alpha=0.8)
    plot.savefig('static/people_photo/first.png')
    

    y, sr = lr.load(audio2)
    ipd.Audio(y, rate=sr)
    plot.figure(figsize=(15, 5))
    lr.display.waveshow(y, sr, alpha=0.8)
    plot.savefig('static/people_photo/first2.png')

def visualize(a1,a2):
    freqdomain(a1,a2)
    oboe(a1,a2)
    spec(a1,a2)
    clarinet(a1,a2)
    first(a1,a2)

