from __future__ import division
import scipy.io.wavfile as wav
import numpy
from mfcc_features import mfcc


with open('words.txt') as f:
    words = f.read().splitlines()

for x in range(len(words)):
    filename = words[x] + "_mfcc"
    mfcc_data = []
    for i in range(10):
        (rate, sig) = wav.read("words_wav/" + words[x] + "-" + str(i + 1) + ".wav")
        print("Reading: " + words[x] + "-" + str(i + 1) + ".wav")
        duration = len(sig) / rate
        mfcc_features = mfcc(sig, rate, winlen=duration / 20, winstep=duration / 20)
        m = mfcc_features[:20]
        mfcc_arr = []
        for i in m:
            mfcc_arr.extend(i)
        mfcc_arr /= numpy.max(numpy.abs(mfcc_arr), axis=0)
        mfcc_data.append(mfcc_arr)

    with open("mfcc/" + filename + ".npy", 'w') as file:
        numpy.save(file, mfcc_data)