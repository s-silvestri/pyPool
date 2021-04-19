#ToDo: Vedere un po' cosa si può fare per i trigger su buche più "dolci", idee di base: allargare la media mobile, controllare la derivata. La mia idea è che le buche molto larghe sono facilmente confondibili con quello che fa il terreno e a quel punto ci sarebbe da crosscheckare con gli indicatori tipici della pavimentazione in cattivo stato. Forse anche contare il numero di picchi che eccedono una certa soglia può fare da trigger.
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
#detection centrata nella buca con mezzo secondo di dati
#dovrai usare pywt.cwt()
windows=False

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
    a=[]
    j=intervallo+1
    a.append(np.average(serie[0:intervallo]))
    while (j<len(serie)-intervallo):
        j+=1
        incr=(serie[j]-serie[j-intervallo])/intervallo
        a.append(a[-1]+incr)
    for j in range (len(serie)-intervallo,len(serie)):
        a.append(np.average(serie[j:len(serie)]))
    return(a)  
    
def rollingstd(serie, intervallo):
    a=[]
    j=intervallo+1
    a.append(np.std(serie[0:intervallo]))
    while (j<len(serie)-intervallo):
        j+=1
        incr=(serie[j]-serie[j-intervallo])/intervallo
        a.append(np.std(serie[j:j+intervallo]))
    for j in range (len(serie)-intervallo,len(serie)):
        a.append(np.average(serie[j:len(serie)]))
    return(a)      

def deviazionemobile(serie,intervallo):
    varianza=[]
    media=[]
    j=intervallo+1
    varianza.append(np.var(serie[0:intervallo]))
    media.append(np.average(serie[0:intervallo]))
    while (j<len(serie)-intervallo):
        j+=1
        incrmu=(serie[j]-serie[j-intervallo])/intervallo
        media.append(media[-1]+incrmu)
        #incrsigma=((serie[j]-media[-2])**2-(serie[j-intervallo]-media[-2])**2)/intervallo
        incrsigma=(serie[j]-media[-2])*(serie[j]-media[-1])-(serie[j-intervallo]-media[-2])*(serie[j-intervallo]-media[-1])
        varianza.append(incrsigma/intervallo)
    for j in range (len(serie)-intervallo,len(serie)):
        media.append(np.average(serie[j:len(serie)]))
        #incrsigma=((serie[j]-media[-2])**2-(serie[j-intervallo]-media[-2])**2)/intervallo
        incrsigma=(serie[j]-media[-2])*(serie[j]-media[-1])-(serie[j-intervallo]-media[-2])*(serie[j-intervallo]-media[-1])
        varianza.append(varianza[-1]+incrsigma/intervallo)
    return (np.sqrt(varianza))
        
def plottamediamobile(wav,campioni):
    mediamobile=rollingavg(wav,campioni)
    xaxis=np.linspace(0,len(mediamobile), num=len(mediamobile))
    plt.plot(xaxis,mediamobile,label='moving average')
    plt.legend()

def plottaserietemporale(serie,campionamento):
    xaxis=np.linspace(0,len(serie)/campionamento, num=len(serie))
    plt.plot(xaxis,serie,label='pressure')
    plt.xlabel('Time [s]')
    plt.xlim(min(xaxis),max(xaxis))
    plt.legend()

def plottaspettrogramma(serie, N, sampling):
    specgramma=plt.specgram(serie,Fs=sampling, window=wind, scale='dB', NFFT=N, cmap='jet')
    plt.yscale('symlog')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    return (specgramma)

def trigger (sample, width, thresh):
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

def filtrabuca (sample, sampling):
    Where=trigger(sample, 4410, 4000)
    buca=[]
    if (Where[0]>0):
        estremi=[Where[1]-math.floor(sampling/2), Where[1]+math.floor(sampling/2)]
        buca=sample[estremi[0]:estremi[1]]*signal.windows.hann(sampling)
    return (buca)

def rollinghole (sample, sampling, width):
    mean=rollingavg(sample**2,width)
    sigma=rollingstd(sample**2,4*width)
    #zeropad sigma
    sigma=np.array(sigma+np.zeros(len(mean)-len(sigma)).tolist())
    print(len(mean))
    print (len(sigma))
    ratio=np.divide(mean,sigma)
    return (np.array(ratio.tolist()+np.zeros(200).tolist()))
    
loadir='/home/kibuzo/Rotoliamo/Dati/misura/'

if windows:
    loadir='C:/Users/acust/Desktop/misura/'
    savedir='c:/Users/acust/Desktop/misura/processed'

filelist=[]
for filename in os.listdir(loadir):
    if filename.endswith(".wav"):
        filelist.append(loadir+filename)

prova=loadwav(filelist[-3])
s=0.01 #(durata in secondi della finestra mobile)
#creo la finestra
N=math.floor(prova[0]*s)
wind=signal.windows.gaussian(N,round(prova[0]/200))
#wind=signal.windows.hann(N)

plt.subplot(313)
ciao=plottaspettrogramma(prova[1], N, prova[0])

plt.subplot(312)
plottaserietemporale(prova[1],prova[0])

plt.subplot (311)
y=rollinghole(prova[1], prova[0], 200).tolist()
x=np.linspace(0,len(y)/prova[0],num=len(y))
plt.plot(x,np.zeros(0).tolist()+y, label='trigger signal')
plt.xlim(0,max(x))
plt.legend()
plt.show()