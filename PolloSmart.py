#Achtung: prima runna pyPool indicando il file giusto da caricare, mi servo di alcune sue funzioni

import numpy
import pandas
from pandas import*
from numpy import *
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Dropout
import seaborn
import tensorflow
import matplotlib.pyplot as plt
import numpy as np

savedir='F:/dataset_navacchio2/'
#Achtung: i savetxt possono dare solamente vettori 2D, pertanto quando lo carichi fai il reshape: c=np.loadtxt(ds.txt), c=np.reshape(c, (np.int(np.shape(b)[0]/16),16,64))
sampling=44100

## Sezione per creare i file e per caricarli
# Creacampione ti restituisce un vettore (time_steps, 64) ovvero (time_steps, features)
def creacampione(data, second=40, time_steps=16):
    irf=psdnorm(data, Fs=sampling, N=4096)
    psdog=psdogramma(data, second, irf, step=0.1, length=time_steps, Fs=sampling)
    return np.array((psdog[0]))

# Per creare una lista di secondi in cui centrare i campioni aumentati
def augment(secondo, numero, time_steps=16):
    return (np.linspace(secondo-0.1*time_steps/2, secondo+0.1*time_steps/2, num=numero))

# crea un dataset a partire da una lista di secondi
def createds(secondi, augmentation=5):
    a=[]
    for j in range (0, len(secondi)):
        augmented=augment(secondi[j], augmentation)
        for i in range (0, len(augmented)):
            a.append(creacampione(prova[1], second=augmented[i]))
    return(a)

def provasecondo(data, second):
    plt.close()
    irf=psdnorm(data, Fs=sampling, N=4096)
    psdog=psdogramma(data, second, irf, step=0.1, length=16, Fs=sampling)
    plt.imshow(np.transpose(psdog[0]))
    plt.colorbar()
    plt.show()

def caricafile(subfoldfile):
    a=np.loadtxt(savedir+subfoldfile)
    return(np.reshape(a, (np.int(np.shape(b)[0]/16),16,64)))

## Sezione LSTM essenzialmente copiata dal tizio online

#Todo: Usa creacampione per campionare i secondi con buche, asfalto bello, asfalto brutto, e possibilmente anche per le buche cerca di capire se sei in asfalto bello o brutto. Crea un dataset accorpandoci anche gli y fatti da vettori (Bello, Brutto, Buca) per tutti i secondi campionati. Comincia a pensare come cambiare prova_lstm per farlo funzionare su questo dataset.
#Todo: taglia dei wav con diversi tipi di distress e classificali nelle cartelle.


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy