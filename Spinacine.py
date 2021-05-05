import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import signal
import math

# Genera un rumore bianco di media mean varianza std lungo seconds (campionato a 44100 Hz)
def genwhite (mean, std, seconds):
    num_samples = 44100
    samples = np.random.normal(mean, std, size=math.floor(seconds*num_samples))
    return (samples)

def genstradabella(s):
    return(genwhite(0,4,s))

# Genera il segnale atteso da una buca che dura seconds secondi, campionato a 44100 Hz
# Per ora è un segnale sinusoidale con finestra gaussiana senza posibilità di variarlo in intensità o in frequenza, magari più in avanti sarà più raffinato man mano che la nostra conoscenza delle buche migliora
def pothole (seconds): # a 40km/h fai 30 cm (tombino quadratico medio) in 0.27 secondi
    samples=math.floor(seconds*44100)
    a=np.linspace(0,20,num=samples)
    b=20*np.sin(100*a)
    wind=signal.windows.gaussian(samples, std=samples/10)
    return(b*wind)
    
#Fa lo zero padding di un vettore: center è la posizione in secondi, length è la durata a cui vogliamo paddare il vettore vector
def zeropad (vector, center, length):
    sampling=44100
    center=center*sampling
    length=length*sampling
    before=math.floor(center-math.floor(len(vector)/2))
    after=length-(len(vector)+before)
    beforev=np.zeros(before)
    afterv=np.zeros(after)
    padded1=np.concatenate ([beforev,vector])
    padded=np.concatenate ([padded1,afterv])
    return (padded)

#Buca una strada: somma buca a strada nella posizione dove in secondi. Si occupa da solo dello zeropadding
def bucastrade (strada, buca, dove):
    stradabucata=strada+zeropad(buca, dove, math.floor(len(strada)/44100))
    xvec=(np.linspace(0,len(strada)/44100, num=len(strada)))
    return (xvec, stradabucata)
    
