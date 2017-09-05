from __future__ import division
import numpy
import math
from scipy.fftpack import dct


def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
         preemph=0.97, ceplifter=22, appendEnergy=True):
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)
    return feat

def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
          preemph=0.97):
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate)
    pspec = powspec(frames, nfft)
    energy = numpy.sum(pspec, 1)
    fb = get_filterbanks(nfilt, nfft, samplerate)
    feat = numpy.dot(pspec, fb.T)
    return feat, energy




def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
             preemph=0.97):
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    return numpy.log(feat)


def ssc(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
        preemph=0.97):
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate)
    pspec = powspec(frames, nfft)
    fb = get_filterbanks(nfilt, nfft, samplerate)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1, samplerate / 2, numpy.size(pspec, 1)), (numpy.size(pspec, 0), 1))
    return numpy.dot(pspec * R, fb.T) / feat


def hz2mel(hz):
    return 2595 * numpy.log10(1 + hz / 700.0)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    bin = numpy.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    fbank = numpy.zeros([nfilt, nfft / 2 + 1])
    for j in xrange(0, nfilt):
        for i in xrange(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in xrange(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def lifter(cepstra, L=22):
    nframes, ncoeff = numpy.shape(cepstra)
    n = numpy.arange(ncoeff)
    lift = 1 + (L / 2) * numpy.sin(numpy.pi * n / L)
    return lift * cepstra


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((1, x))):
    slen = len(sig)
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + math.ceil((1.0 * slen - frame_len) / frame_step)

    padlen = (numframes - 1) * frame_step + frame_len

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len), (numframes, 1))
    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.ones((1, x))):
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((1, padlen))
    window_correction = numpy.zeros((1, padlen))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])