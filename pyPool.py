#ToDo: Stimare la larghezza delle due finestre mobili per il trigger, una per la media fuori buca che deve rimanere tale anche dentro la buca (quindi 10 volte più larga della tipica buca) e una che è una correzione della media puntuale anche dentro la buca, quindi più stretta. La varianza andrebbe fatta fuori buca, quindi sul campione mobile, FAI LA VARIANZA MOBILE.
#Achtung: le tue medie mobili sembrano produrre in uscita vettori che sono lunghi almeno 2 volte la lunghezza della finestra, il che credo che abbia senso ma non ho tempo di controllarlo, nel dubbio usalo solo su vettori lunghi
from scipy.io.wavfile import read
from scipy.signal import spectrogram
import os
import matplotlib.pyplot as plt
import numpy as np
import acoustics.signal as ac
from scipy import signal
import math
import pywt
from time import time
from matplotlib.colors import LogNorm
import pandas as pd
#detection centrata nella buca con mezzo secondo di dati
#dovrai usare pywt.cwt()
windows=False

@plt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%2d}$' % (x)

#loadwav carica file e restituisce un vettore dove il primo elemento è la frequenza di sampling, il secondo è la serie numerica
def derivata(serie):
    der=[]
    der.append(serie[1]-serie[0])
    for j in range (1, len(serie)-1):
        der.append(serie[j+1]-serie[j])
    return (np.array(der))

def loadwav(file):
    samplerate, data = read(file)
    return (samplerate, data*813.5330/32767) # quel numerino è per la calibrazione del sensore audio

#La vecchia media mobile: lenta ma testata
def moveaverag(serie, intervallo):
    a=[]
    for j in range (0,len(serie)-intervallo):
        a.append(np.average(serie[j:j+intervallo]))
    return(a)

def rollingavg(serie, intervallo):
    a=np.zeros(intervallo).tolist()
    j=intervallo
    a.append(np.average(serie[0:intervallo]))
    while (j<len(serie)-intervallo):
        j+=1
        incr=(serie[j]-serie[j-intervallo])/intervallo
        a.append(a[-1]+incr)
    for j in range (len(serie)-intervallo,len(serie)-1):
        a.append(np.average(serie[j:len(serie)]))
    return(a)

def rollingavgnonzero(serie, intervallo):
    a=np.zeros(intervallo).tolist()
    j=intervallo
    a.append(np.average(serie[0:intervallo]))
    while (j<len(serie)-intervallo):
        j+=1
        incr=(serie[j]-serie[j-intervallo])/intervallo
        appendor=(a[-1]+incr)
        if (appendor>0):
            a.append(a[-1]+incr)
        else:
            a.append(a[-1])
    for j in range (len(serie)-intervallo,len(serie)-1):
        appendor=np.average(serie[j:len(serie)])
        if (appendor>0):
            a.append(appendor)
        else:
            a.append(a[-1])
    return(a)

def rollingstd(serie, intervallo):
    a=[]
    j=intervallo+1
    a.append(np.std(serie[0:intervallo]))
    while (j<len(serie)-intervallo):
        j+=1
        a.append(np.std(serie[j:j+intervallo]))
    for j in range (len(serie)-intervallo,len(serie)):
        a.append(np.average(serie[j:len(serie)]))
    return(a)

# def deviazionemobile(serie,intervallo):
#     varianza=[]
#     media=[]
#     j=intervallo+1
#     varianza.append(np.var(serie[0:intervallo]))
#     media.append(np.average(serie[0:intervallo]))
#     print(len(media))
#     print (len (varianza))
#     while (j<len(serie)-intervallo):
#         intesimo=1/intervallo
#         j+=1
#         next=j
#         curr=j-1
#         zero=(curr-intervallo)
#         incrmu=(serie[next]-serie[zero])/intervallo
#         media.append(media[-1]+incrmu)
#         primofattore=intesimo*(serie[next]-serie[zero])**2
#         secondofattore=(serie[next]-media[-1])**2
#         terzofattore=(serie[zero]-media[-1])**2
#         quartofattore=2*intesimo*(serie[next]-serie[zero]*np.sqrt(np.abs(varianza[-1]))
#         incrsigma=(primofattore+secondofattore-terzofattore-quartofattore)
#         varianza.append(varianza[-1]+incrsigma)
#     for j in range (len(serie)-intervallo,len(serie)-1):
#         j+=1
#         intervallo-=1
#         intesimo=1/intervallo
#         next=j
#         curr=j-1
#         zero=(curr-intervallo)
#         incrmu=(serie[next]-serie[zero])/intervallo
#         media.append(media[-1]+incrmu)
#         primofattore=intesimo*(serie[next]-serie[zero])**2
#         secondofattore=(serie[next]-media[-1])**2
#         terzofattore=(serie[zero]-media[-1])**2
#         quartofattore=2*intesimo*(serie[next]-serie[zero]*np.sqrt(varianza[-1]))
#         incrsigma=(primofattore+secondofattore-terzofattore-quartofattore)
#         varianza.append(varianza[-1]+incrsigma)
#     return (np.sqrt(varianza))

def plottamediamobile(wav,campioni):
    mediamobile=rollingavg(wav,campioni)
    xaxis=np.linspace(0,len(mediamobile), num=len(mediamobile))
    plt.plot(xaxis,mediamobile,label='moving average')
    plt.legend()

def plottaserietemporale(serie,campionamento):
    xaxis=np.linspace(0,len(serie)/campionamento, num=len(serie))
    plt.plot(xaxis,serie,label='Pressione [RMS]')
    plt.xlabel('Time [s]')
    plt.xlim(min(xaxis),max(xaxis))
    plt.legend()

def plottaspettrogramma(serie, N, sampling):
    specgramma=plt.specgram(serie,Fs=sampling, window=wind, scale='dB', NFFT=N, cmap='jet')
    plt.yscale('symlog')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    return (specgramma)

def triggerold (sample, width, thresh):
    tic=time()
    for j in range (0,len(sample)-width):
        Trigger=0
        if(np.average(sample[j:j+width]**2)>thresh):
            Trigger=1
            break
    toc=time()
    print ('Tempo di esecuzione per ' + str(j) + ' cicli di esecuzione del trigger= ' + str(toc-tic)+ ' s')
    print ('Tempo medio di calcolo del trigger: ' + str(1e6*(toc-tic)/j) + ' us')
    return (Trigger,j)

def trigger (sample, width, thresh):
    tic=time()
    for j in range (0,len(sample)-width):
        Trigger=0
        if max (rollinghole(sample[j:j+width], 44100, 200)>30):
            Trigger=1
            break
    toc=time()
    print ('Tempo di esecuzione per ' + str(j) + ' cicli di esecuzione del trigger= ' + str(toc-tic)+ ' s')
    print ('Tempo medio di calcolo del trigger: ' + str(1e6*(toc-tic)/j) + ' us')
    return (Trigger,j)

def filtrabuca (sample, sampling):
    Where=triggerold(sample, 4410, 1000)
    buca=[]
    if (Where[0]>0):
        estremi=[Where[1]-math.floor(sampling/2), Where[1]+math.floor(sampling/2)]
        buca=sample[estremi[0]:estremi[1]]*signal.windows.hann(sampling)
    return (buca)


def rollinghole (sample, sampling, width):
    mean=rollingavg(sample**2,width)
    sigma=rollingstd(mean,1*width)
    #zeropad sigma
    sigma=np.array(sigma+np.zeros(len(mean)-len(sigma)).tolist())
    #print(len(mean))
    #print (len(sigma))
    ratio=np.divide(mean[100:],sigma[:-100])
    return (np.array(ratio.tolist()+np.zeros(200).tolist()))

#Fa la trasformata di wavelet suddividendo gli intervalli di frequenze in nlogbin, da migliorare.
def wavelet (sig, nlogbin):
    fsampling=44100
    #dt=0.01
    logextent=[1,4]
    #frequencies = pywt.scale2frequency('cmor1.5-1.0',np.arange(1,nlogbin)) / 0.1
    widths=np.logspace(logextent[0],logextent[1],num=nlogbin)
    cwtmatr, freqs = pywt.cwt(sig, widths, 'cmorl3-1')
    fig, (ax) = plt.subplots()
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_formatter(fake_log)
    imm=ax.imshow(np.abs(cwtmatr), aspect='auto', cmap='jet', vmax=abs(cwtmatr).max(), vmin=abs(cwtmatr).min(), extent=[0,len(sig)/fsampling,logextent[0], logextent[1]])  # doctest: +SKIP
    ax.set_ylabel('frequency [Hz]')
    ax.set_xlabel('time [s]')
    fig.colorbar(imm)
    plt.show()

def triggernew(segnale, largewindow, smallwindow):
    sampling=44100
    rollinglarge=(rollingavgnonzero(segnale**2, 10000))
    rollingsmall=(rollingavg(segnale**2, 200))#[(largewindow-smallwindow):])
    nsecutili=math.floor(len(rollingsmall)/sampling)#[:-(largewindow-smallwindow)])/sampling)
    rollinglargeaffettato=rollinglarge[:nsecutili*sampling]
    rollingsmallaffettato=rollingsmall[:nsecutili*sampling]
    trigger=np.divide(rollingsmallaffettato,rollinglargeaffettato)
    xtime=np.linspace(0, len(rollinglargeaffettato)/44100, num=len(rollinglargeaffettato))
    return(xtime, trigger, rollingsmallaffettato, rollinglargeaffettato)



loadir='/home/kibuzo/Rotoliamo/Dati/misura/'

if windows:
    loadir='C:/Users/acust/Desktop/misura/Longnavacchio/'
    savedir='c:/Users/acust/Desktop/misura/processed'

filelist=[]
for filename in os.listdir(loadir):
    if filename.endswith(".wav"):
        filelist.append(loadir+filename)


# trigger=[]
# rollinglarge=(rollingavg(prova[1]**2, 10000))
# rollingsmall=(rollingavg(prova[1]**2, 200)[(10000-200):])
# for j in (0,len(prova[1])-10200):
#     trigger.append(rollingsmall[-1]/rollinglarge[-1])
#     nsec=math.floor(len(rollinglarge[200:-10000])/44100)
#     nsecutili=math.floor(len(rollingsmall[800:])/44100)
#     rollinglargeaffettato=rollinglarge[:nsecutili*44100]
#     rollingsmallaffettato=rollingsmall[:nsecutili*44100]
#     trigger=np.divide(rollingsmallaffettato,rollinglargeaffettato)
#     plt.plot(np.linspace(0,len(rollinglargeaffettato)/44100, num=len(rollinglargeaffettato)),np.divide(rollingsmallaffettato,rollinglargeaffettato))
def triplot (segnale,sampling):
    Trigger=triggernew(segnale, 10000,200)
    plt.subplot(311)
    plottaserietemporale(np.sqrt(segnale**2),sampling)
    plt.subplot(312)
    plt.plot(Trigger[0],Trigger[2], label='stima puntuale della pressione rms')
    plt.plot(Trigger[0],Trigger[3], label='stima del background rms')
    plt.xlim(0, max(Trigger[0]))
    plt.legend()
    plt.subplot(313)
    plt.plot(Trigger[0], Trigger[1], label='segnale del trigger')
    plt.xlim(0,max(Trigger[0]))
    plt.legend()
    plt.show()
    

prova=loadwav(filelist[8])
# Triggah=triggernew(prova[1], 10000, 200)
# s=0.01 #(durata in secondi della finestra mobile)
# #creo la finestra
# N=math.floor(prova[0]*s)
# wind=signal.windows.gaussian(N,round(prova[0]/200))
# #wind=signal.windows.hann(N)
# 
# plt.subplot(313)
# plt.plot(Triggah[0], Triggah[1])
# 
# plt.subplot(312)
# plottaserietemporale(np.sqrt(prova[1]**2),prova[0])
# 
# 
# plt.subplot (311)
# plt.plot(Triggah[0],Triggah[2], label='stima puntuale della pressione rms')
# plt.plot(Triggah[0],Triggah[3], label='stima del background rms')
# plt.legend()
# plt.show()