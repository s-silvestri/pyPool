#Achtung: le tue medie mobili sembrano produrre in uscita vettori che sono lunghi almeno 2 volte la lunghezza della finestra, il che credo che abbia senso ma non ho tempo di controllarlo, nel dubbio usalo solo su vettori lunghi
from scipy.io.wavfile import read
from scipy.io.wavfile import write
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
import scipy as sp
import scipy.fftpack
from spectrum import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
from scipy.signal import butter, lfilter, freqz
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from multiprocessing import Pool
import tensorflow as tf
from scipy.io import wavfile
import soundfile as sf
windows=True
#loadwav carica file e restituisce un vettore dove il primo elemento è la frequenza di sampling, il secondo è la serie numerica. Il numero di conversione serve per passare da canali adc a pressione (canali adc 32767 e pressione in pascal 813.5330)
def loadwav(file):
    samplerate, data = read(file)
    #Definisco il fattore di scala della traccia audio. 32767 sono i canali adc da convertire in [-1,1] per un 16 bit, mentre 813 è per passare in pressione. Se usassi tensorflow o matlab basta 813
    scale=813.5330/32767
    return (samplerate, data*scale) 

#---------------------- Qui ci sono un po' di indirizzi di cartelle da cui volendo si caricano facilmente i file: ognuno le aggiorni con le sue folder preferite, idealmente tutti i file sono in una cartella poi vengono caricate in un vettore filelist[].
#loadir='/media/kibuzo/Gatto Silvestro/Buche/Misure Coltano/pint/unzipped/run_spezzate/'
loadir='/media/kibuzo/6317E4611EC4DD5F/Buche/misure_a_giro_ospedaletto_2/Pint/unzipped/run_spezzate/'

savedir='/media/kibuzo/Gatto Silvestro/Buche/Misure navacchio/gopro/00082/triggered/plot/'
savedir='/media/kibuzo/Gatto Silvestro/Buche/Misure Coltano/pint/unzipped/run_spezzate/Untriggered/Gronchi/'


if windows:
    #loadir='C:/Users/acust/Desktop/misura/Long/'
    #loadir='F:/Misure Coltano 2/pint/unzipped/Run spezzate/run/run1/'
    #loadir='F:/pint/Misure Coltano 2/pint/unzipped/Run spezzate/run/run7/'
    loadir='F:/pint/Misure Coltano/pint/unzipped/run_spezzate/'
    #loadir='E:/Buche\Misure Coltano/pint/unzipped/run_spezzate/'
    #loadir='F:/Misure navacchio 2/pint/unzipped/run_spezzate/'
    loadirph='F:/buche30/'
    heihei='F:/Reggio/PIPPO_FUORI_ALBERGO/wav/'
    savedir='c:/Users/acust/Desktop/misura/processed/'

#Questo mette tutti i nomi dei file in un apposito vettore.
polli=[]
for filename in os.listdir(heihei):
    if filename.endswith(".wav"):
        polli.append(heihei+filename)
pollo=sf.read(polli[-1])
pollo=np.array(pollo).tolist()
pollo[0]*=813.5330
pollo=np.array(pollo)

filelist=[]
for filename in os.listdir(loadir):
    if filename.endswith(".wav"):
        filelist.append(loadir+filename)
prova=loadwav(filelist[-1])
#test=loadwav(filelist[-3])

filelistph=[]
for filename in os.listdir(loadirph):
    if filename.endswith(".wav"):
        filelistph.append(loadirph+filename)
    
#provabrutto=loadwav(filelist[2])
#encoder=pd.read_csv(loadir+'vel_secondo.csv')

#---------------------- Cominciano le funzioni

#Copiato da internet. Usato, e poi abbandonato, per l'asse verticale logaritmico della wavelet: ha diversi problemi, tipo che ti approssima 5x10^23 in 10^23, magari un giorno lo aggiusto
@plt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%2d}$' % (x)

#funzione seno usata per il fit
def sin(amplitude, phase, angular, t):
    return amplitude*np.sin(angular*t+phase)

def kph(mps):
    return (3.6*mps)
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def trimMean(tlist,tperc):
    removeN = int(math.floor(len(tlist) * tperc / 2))
    tlist.sort()
    if removeN > 0: tlist = tlist[removeN:-removeN]
    return reduce(lambda a,b : a+b, tlist) / float(len(tlist))
    
#Le prossime funzioni trasformano l'input in secondi nel numero di campioni e viceversa. Da aggiornare se cambia la frequenza di sampling    
def secondi(campioni):
    sampling=44100
    return (campioni/sampling)
    
def campioni(secondi):
    sampling=44100
    return (math.floor(secondi*sampling))

def PtoLeq(P):
    P0=2e-5
    return (10*np.log10(P**2/P0**2))
    
def fittaseno(data):
    pdata=pd.Series(data)
    upsampled = pdata.resample('D')
    interpolated = upsampled.interpolate(method='spline', order=2)
    x=np.linspace(0,len(data)-1, len(data))
    from scipy.optimize import curve_fit
    popt,pcov=curve_fit(sin,x,data,p0=(np.max(data), data[0]/np.max(data), 0.014))
    return popt,x
        
    
#prendo tutto in secondi, per ora assumo comportamento virtuoso dell'operatore, cioè le buche non devono essere sul bordo (intervallo di mezzo secondo), né fuori. Metti anche un check sulle buche, se mi passa un array vuoto non devo fare niente.
def toglibuche(serie, inizio, fine, buche):
    xvec=np.linspace(inizio,fine,num=math.floor((fine-inizio)*44100))
    if np.array(buche).size==0:
        return(xvec, serie)
    buche=np.array(buche)
    serie=serie[campioni(inizio):campioni(fine)]
    buche=buche-inizio
    #aggiungere un controllo su buca prima di inizio serie
    s=(buche[0]-0.5)
    N=math.floor(44100*s)
    #wind=signal.windows.hann(N)
    wind=1
    senzabuche=serie[0:campioni(buche[0]-0.5)]
    xtime=xvec[0:campioni(buche[0]-0.5)]
    #print(len(senzabuche))
    #print(len(xtime))
    for j in range (1,len(buche)):
        #Il segnale buono sta tra inf e sup, mentre le buca  tra sup e sup+1sec e tra inf e inf-1sec
        inf=buche[j-1]+0.5
        sup=buche[j]-0.5
        infC=campioni(inf)
        supC=campioni(sup)
        s=sup-inf
        #wind=signal.windows.hann(N)
        wind=1
        N=math.floor(44100*s)
        senzabuche=np.concatenate((senzabuche,serie[infC:supC]*wind))
        print ('removing hole number at second '+ str(inf+0.5))
        xtime=np.concatenate((xtime,xvec[infC:supC]))
    inf=buche[-1]+1
    infC=campioni(inf)
    supC=campioni(fine)
    senzabuche=np.concatenate((senzabuche,serie[infC:]))
    xtime=np.concatenate((xtime,xvec[infC:]))
    print ('Returned vectors in form of (time, signal), the second is hole-free')
    return(xtime,senzabuche)
    
#Funzione sperimentale che prende una serie temporale in pressione, toglie le buche segnate nel vettore buchesecondi assumendo una durata di un secondo centrato nel timestamp e calcola il leq del risultato. L'incertezza sarà calcolata come RMS/sqrt(BT) a cui sommare la varianza dei livelli, al momento il primo contributo l'ho tolto. 
def meanLeq(serie, bandwidth):
    #La dimensione del campione è idealmente almeno un decimo di secondo, ma voglio almeno 30 campioni in tutto
    samplesize=math.floor(np.minimum(len(serie)/30, 441))
    #RMS=(np.sqrt(np.average((serie)**2)))
    #gli intervalli partono da 0 e arrivano a tutta la dimensione del campione, in step samplesize
    rms=[]
    j=0
    while j<len(serie)-samplesize:
        j+=samplesize
        rms.append(np.sqrt(np.mean(serie[j:j+samplesize]**2)))
    # for j in range (0, len(intervals)):
    #     rms.append(np.sqrt(np.sum(serie[j:j+samplesize]**2)))
    #print (j/samplesize)
    rms=np.array(rms)
    #sigma2=RMS/(np.sqrt(bandwidth*secondi(len(serie))))
    sigma= np.sqrt(np.var(rms))
    RMS=np.median(rms)
    RMSup=RMS+sigma
    RMSdown=RMS-sigma
    return(PtoLeq(RMS), PtoLeq(RMSup), PtoLeq(RMSdown))
    
#calcola il leq e gli intervalli di confidenza per una strada con buche a cui toglie le buche
def calcoleq(strada, intervallo):
    strada=strada[campioni(intervallo[0]):campioni(intervallo[1])]
    trigger=markholes(ac.octavepass(center=40, fs=44100, fraction=3, signal=strada, order=8), 44100, 40, 0)
    sbucato=toglibuche(strada, 0, secondi(len(strada)), trigger)
    return (meanLeq(sbucato[1], 10))

def sbucastrada(strada,intervallo):
    strada=strada[campioni(intervallo[0]):campioni(intervallo[1])]
    trigger=markholes(ac.octavepass(center=40, fs=44100, fraction=3, signal=strada, order=8), 44100, 40, 0)
    sbucato=toglibuche(strada, 0, secondi(len(strada)), trigger)
    return(sbucato)
    
 
#Le seguenti servono per le medie mobili: per la media mobile funziona sia quella di default di pandas sia quella scritta da me che lascio soprattutto per motivi di retrocompatibilità e confronto
def rollingavgpd(serie, intervallo):
    pdserie=pd.Series(serie)
    avg=pdserie.rolling(intervallo).mean()
    return (avg.to_numpy())
    
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

def rollingavgnonzero(serie, intervallo, media200):
    init=np.average(serie[0:intervallo])
    a=[]
    trigUP=[]
    for j in range (0,intervallo):
        a.append(init)
    while (j<len(serie)-intervallo):
        j+=1
        if(media200[j-1]>1.5*a[j-1]):
            trigUP.append(j) 
            a.append(a[-j])
            #devo salvarmi questo intervallo e ricordarmi di escluderlo dall'appendor
        else:
            if trigUP and j in range (min(trigUP)+intervallo, max(trigUP)+intervallo):
                incr=(serie[j]-serie[j-2*intervallo])/intervallo
                if (j-intervallo)==max(trigUP): 
                    trigUP=[]
            else:
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

#La più abusata (con pandas) fino a maggio: restituisce medie mobili a pochi e tanti campioni, varianza su tanti campioni. Il trigger si calcola banalmente da x-mu/sigma.
def rollingtrig(serie, lungo, corto):
    seriepd=pd.Series(serie)
    largemean=seriepd.rolling(lungo).median()
    largevar=seriepd.rolling(lungo).var()
    smallmean=seriepd.rolling(corto).median()
    return (smallmean.to_numpy(), largemean.to_numpy(), largevar.to_numpy())

def terzottava(segnale, centrobanda):
    #centeroctave=1000*2.**(np.arange(-7,7))
    sampling=44100
    filtrato=ac.octavepass(center=centrobanda, fs=sampling, fraction=3, signal=segnale, order=8)
    return (filtrato)

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
    plt.legend(loc='upper right')

def plottaspettrogramma(serie, N, sampling):
    specgramma=plt.specgram(serie,Fs=sampling, scale='dB', NFFT=N, cmap='jet')
    plt.yscale('symlog')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    return (specgramma)

def plottaspettro(data):
    sampling=44100
    # Number of samplepoints
    N = len(data)
    # sample spacing
    T = 1.0 / sampling
    x = np.linspace(0.0, float(N/sampling), num=N)
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = scipy.fftpack.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), math.floor(N/2))
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlim(-100,5000)
    plt.xlabel('Frequency[Hz]')
    plt.show()

def plottaspettrosbucato(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    data=sbucastrada(data,(intervallo[0],intervallo[1]))[1]
    # Number of samplepoints
    N = len(data)
    # sample spacing
    T = 1.0 / sampling
    x = np.linspace(0.0, float(N/sampling), num=N)
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = scipy.fftpack.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), math.floor(N/2))
    fig, ax = plt.subplots()
    ax.semilogx(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlim(100,5000)
    plt.xlabel('Frequency[Hz]')
    plt.show()

def plottapsdsbucata(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    data=sbucastrada(data,(intervallo[0],intervallo[1]))[1]
    # Number of samplepoints
    N = len(data)
    # sample spacing
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    #fig, ax = plt.subplots()
    N=4096#8192
    wind=sp.signal.blackmanharris(N)
    a=plt.psd(data, NFFT=N, Fs=44100, window=wind, detrend=None)
    plt.xscale('log')
    #ax.semilogx(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlim(100,12800)
    #plt.xlim(1,100)
    plt.xlabel('Frequency[Hz]')
    plt.show()
    return (a)
    
def plottapsdbucata(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    data=data[campioni(intervallo[0]):campioni(intervallo[1])]
    # Number of samplepoints
    N = len(data)
    # sample spacing
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    #fig, ax = plt.subplots()
    N=4096#8192
    wind=sp.signal.blackmanharris(N)
    a=plt.psd(data, NFFT=N, Fs=44100, window=wind, detrend=None)
    plt.xscale('log')
    #ax.semilogx(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlim(100,12800)
    #plt.xlim(1,100)
    plt.xlabel('Frequency[Hz]')
    return (a)
    

#Plotta l'autoregressione yule walker nell'intervallo specificato in secondi, da indicare come vettore
def plottayule(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    wind=sp.signal.blackmanharris(campioni(intervallo[1])-campioni(intervallo[0]))
    #data=sbucastrada(data,(intervallo[0],intervallo[1]))[1]
    data=data[campioni(intervallo[0]):campioni(intervallo[1])]
    N = 8192
    order=500
    p = pyule(data*wind, order, NFFT=N, sampling=44100)
    #psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
    p()
    p.plot(sides='centerdc')
    plt.xscale('log')
    plt.xlim(100,12800)
    return (p)
    #plt.xlim(1,20)
    #plt.show()
    
def getyule(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    wind=sp.signal.blackmanharris(campioni(intervallo[1])-campioni(intervallo[0]))
    #data=sbucastrada(data,(intervallo[0],intervallo[1]))[1]
    data=data[campioni(intervallo[0]):campioni(intervallo[1])]
    N = 8192
    order=500
    p = pyule(data*wind, order, NFFT=N, sampling=44100)
    #psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
    p()
    return (p)
    #plt.xlim(1,20)
    #plt.show()


def poweryule(data):
    a=getyule(data, (0,secondi(len(data))))
    psd=a.psd
    #vettore delle frequenze
    frequenze=np.array(a.frequencies())
    spaziatura=(frequenze[1:]-frequenze[:-1])[-1]
    #comincio a tagliare in frequenza
    integrale=np.sum(spaziatura*psd)
    return 2*integrale
    
# Calcola l'integrale della PSD in un intervallo di frequenza. Ricorda che freq2 è almeno la frequenza di nyquist, cioè 22000.
def powerbandyw(data, freq1, freq2):
    a=getyule(data, (0,secondi(len(data))))
    psd=a.psd
    #vettore delle frequenze
    frequenze=np.array(a.frequencies())
    spaziatura=(frequenze[1:]-frequenze[:-1])[-1]
    #comincio a tagliare in frequenza
    cutbasso=frequenze[frequenze>freq1]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*psd[indicebasso:indicealto])
    return 2*integrale
    
def poweratio(data, freq1, freq2, freq3, freq4):
    a=getyule(data, (0,secondi(len(data))))
    Psd=a.psd
    #vettore delle frequenze
    frequenze=np.array(a.frequencies())
    spaziatura=(frequenze[1:]-frequenze[:-1])[-1]
    #comincio a tagliare in frequenza
    cutbasso=frequenze[frequenze>freq1]
    cutbasso1=frequenze[frequenze>freq3]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    indicebasso1=np.where(frequenze>cutbasso1[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    band1=cutbasso1[cutbasso1<freq4]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    indicealto1=(np.where(frequenze<band1[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*Psd[indicebasso:indicealto])
    integrale1=np.sum(spaziatura*Psd[indicebasso1:indicealto1])
    return integrale/integrale1
    
#Fa la cwt del segnale. Non superare i 2 secondi e non superare i 20 bin logaritmici. Mi raccomando di fare attenzione alla scala verticale che è brutta perché una logaritmica artificiale in base 10, cioè sull'asse leggi il valore dell'esponente
def wavelet (sig, nlogbin):
    fsampling=44100
    widths=np.logspace(np.log10(fsampling/5000),np.log10(fsampling/10), num=nlogbin)
    cwtmatr, freqs = pywt.cwt(sig, widths, 'cmorl3-1')
    extent=[(np.min(1/(widths/44100))), (np.max(1/(widths/44100)))]
    logextent=np.log10(extent)
    fig, (ax) = plt.subplots()
    ax.yaxis.set_major_locator(plt.MaxNLocator(len(widths)))
    #ax.yaxis.set_major_formatter(fake_log)
    imm=ax.imshow(np.abs(cwtmatr), aspect='auto', cmap='jet', vmax=abs(cwtmatr).max(), vmin=abs(cwtmatr).min(), extent=[0,len(sig)/fsampling, logextent[0], logextent[1]]) 
    ax.set_ylabel('Log frequency')
    ax.set_xlabel('time [s]')
    fig.colorbar(imm)

# funzione che marca le buche, restituisce timestamp, rms del segnale, deviazione standard del fondo. richiede segnale, frequenza di sampling e differenza tra il tempo nell'audio e nel video. Achtung: adesso ho messo una roba che toglie l'eccesso di varianza per un po' di sample dopo il trigger
def markholes(segnale, sampling, centerband, deltat):
    soglia=3
    longroll=math.floor((sampling*25)/centerband)
    shortroll=math.floor(sampling/(2*centerband))
    trig=rollingtrig(np.sqrt(segnale**2), longroll, shortroll)
    #varianza=np.array(trig[2])
    #varianzaback=np.concatenate((varianza[longroll:],np.zeros(longroll)))
    z=np.sqrt((trig[0]-trig[1])**2/trig[2])
    trigUP=z>soglia
    #varianza=varianza-(varianza-varianzaback)*trigUP
    timestamp=[]
    x=[]
    mu=[]
    sigma=[]
    j=0
    while (j< len(trig[1])):
        if (z[j])>soglia:
            #if (trig[1][j]>1):
            #    if (trig[2][j]>1):
            timestamp.append(j/sampling-deltat)
            #index=np.where(z==np.max(z[j:j+441]))
            #x.append(float(trig[0][index]))
            #mu.append(float(trig[1][index]))
            #sigma.append(float(trig[2][index]))
            #sigma.append(varianza[index])
            #Se il trigger è alto skippa un secondo, per evitare di avere tanti positivi
            j+=sampling-1
        j+=1
    return(timestamp)#,x,mu,np.sqrt(sigma))

#Questo fa 3 plot utili impilati verticalmente del segnale in pressione dato in input. Ormai dà per scontato che il segnale sia una banda in terzi d'ottava centrata in centerband
def triplot (segnale,sampling, centerband):
    longroll=math.floor((sampling*25)/centerband)
    shortroll=math.floor(sampling/(2*centerband))
    Mediemobili=rollingtrig(np.sqrt(segnale**2), longroll,shortroll)
    z=np.sqrt((Mediemobili[0]-Mediemobili[1])**2/Mediemobili[2])
    # trigUP=np.zeros(len(Mediemobili[2]))
    # dove=np.where(z>4)
    # for j in range (0,len(dove[0])):
    #     trigUP[dove[0][j]-math.floor(0.2*longroll):dove[0][j]+math.floor(1.2*longroll)]=1
    # print (np.sum(trigUP))
    # varianza=Mediemobili[2]
    # varianzaback=np.concatenate((np.zeros(2*longroll),varianza[:-2*longroll]))
    # varianza=varianza-(varianza-varianzaback)*trigUP
    # Ritrigger=np.sqrt((Mediemobili[0]-Mediemobili[1])**2/varianza)
    xvec=np.linspace(0, len(Mediemobili[0])/sampling, num=len(Mediemobili[0]))
    plt.subplot(311)
    plottaserietemporale(np.sqrt(segnale**2),sampling)
    plt.subplot(312)
    plt.plot(xvec,Mediemobili[0], label='stima puntuale della pressione rms')
    plt.plot(xvec,Mediemobili[1], label='stima del background rms')
    plt.plot(xvec,np.sqrt(Mediemobili[2]), label='stima della varianza background')
    #plt.plot(xvec,np.concatenate((np.zeros(1),400*np.diff((Mediemobili[0])))), label='derivata di rms')
    #plt.plot(varianza-varianzaback, label='bamboo')
    plt.xlim(0, max(xvec))
    plt.legend(loc='upper left')
    plt.subplot(313)
    triggers=np.sqrt(((Mediemobili[0]-Mediemobili[1])**2)/Mediemobili[2])
    plt.plot(xvec, z, label='segnale del trigger')
    #plt.plot(xvec, Ritrigger, label='segnale del trigger pulito')    
    print(str(np.sum(triggers>5)) + ' samples triggered')
    plt.ylim(1,7)
    plt.xlim(0,max(xvec))
    plt.axhline(y=4,xmin=0, xmax=max(xvec), linestyle='--')
    plt.legend(loc='upper left')
    plt.show()

#------------- Qui cominciano le funzioni che salvano automaticamente i file nella cartella savedir
def savecwt(segnale, deltat, savedir, timestamps):
    for j in range (0, len(timestamps)):
        intervallo=[campioni((timestamps[j]+deltat-1)),campioni((timestamps[j]+deltat+1))]
        wavelet(segnale[intervallo[0]:intervallo[1]],20)
        plt.savefig(savedir+str(int(timestamps[j]))+'_cwt')
        plt.close()

def saveyw(segnale, deltat, savedir, timestamps):
    for j in range (0, len(timestamps)):
        intervallo=((timestamps[j]+deltat-1),(timestamps[j]+deltat+1))
        plottayule(segnale,(intervallo[0],intervallo[1]))
        plt.savefig(savedir+str(int(timestamps[j]))+'_YW_psd')
        plt.close()

def savetriplot(segnale, deltat, savedir, timestamps):
    sampling=44100
    Mediemobili=rollingtrig(np.sqrt(segnale**2), 10000,200)
    xvec=np.linspace(0, len(Mediemobili[0])/sampling, num=len(Mediemobili[0]))
    for j in range (0, len(timestamps)):
        print(timestamps[j])
        intervallo=(math.floor((timestamps[j]+deltat-1)*44100),math.floor((timestamps[j]+deltat+1)*44100))
        signal=segnale[intervallo[0]:intervallo[1]]
        plt.subplot(311)
        plottaserietemporale(np.sqrt(signal**2),sampling)
        plt.subplot(312)
        plt.plot(xvec[intervallo[0]:intervallo[1]],Mediemobili[0][intervallo[0]:intervallo[1]], label='pressione rms')
        plt.plot(xvec[intervallo[0]:intervallo[1]],Mediemobili[1][intervallo[0]:intervallo[1]], label='fondo rms')
        plt.plot(xvec[intervallo[0]:intervallo[1]],np.sqrt(Mediemobili[2][intervallo[0]:intervallo[1]]), label='varianza')
        plt.xlim(min(xvec[intervallo[0]:intervallo[1]]), max(xvec[intervallo[0]:intervallo[1]]))
        plt.legend(loc='upper left')
        plt.subplot(313)
        plt.plot(xvec[intervallo[0]:intervallo[1]], np.sqrt(((Mediemobili[0][intervallo[0]:intervallo[1]]-Mediemobili[1][intervallo[0]:intervallo[1]])**2)/Mediemobili[2][intervallo[0]:intervallo[1]]), label='trigger')
        plt.xlim(min(xvec[intervallo[0]:intervallo[1]]),max(xvec[intervallo[0]:intervallo[1]]))
        #plt.legend(loc='upper left')
        #plt.show()
        plt.savefig(savedir+str(int(timestamps[j]))+'_triplot')
        plt.close()

def savetriplotresonant(segnale, deltat, savedir, timestamps, centerband):
    sampling=44100
    Mediemobili=rollingtrig(np.sqrt(ac.octavepass(segnale,centerband, fs=sampling, order=8, fraction=3)**2), math.floor((sampling*25)/centerband),math.floor(sampling/(2*centerband)))
    xvec=np.linspace(0, len(Mediemobili[0])/sampling, num=len(Mediemobili[0]))
    for j in range (0, len(timestamps)):
        print(timestamps[j])
        intervallo=(math.floor((timestamps[j]+deltat-1)*44100),math.floor((timestamps[j]+deltat+1)*44100))
        signal=segnale[intervallo[0]:intervallo[1]]
        plt.subplot(311)
        plottaserietemporale(np.sqrt(signal**2),sampling)
        plt.subplot(312)
        plt.plot(xvec[intervallo[0]:intervallo[1]],Mediemobili[0][intervallo[0]:intervallo[1]], label='pressione rms')
        plt.plot(xvec[intervallo[0]:intervallo[1]],Mediemobili[1][intervallo[0]:intervallo[1]], label='fondo rms')
        plt.plot(xvec[intervallo[0]:intervallo[1]],np.sqrt(Mediemobili[2][intervallo[0]:intervallo[1]]), label='varianza')
        plt.xlim(min(xvec[intervallo[0]:intervallo[1]]), max(xvec[intervallo[0]:intervallo[1]]))
        plt.legend(loc='upper left')
        plt.subplot(313)
        plt.plot(xvec[intervallo[0]:intervallo[1]], np.sqrt(((Mediemobili[0][intervallo[0]:intervallo[1]]-Mediemobili[1][intervallo[0]:intervallo[1]])**2)/Mediemobili[2][intervallo[0]:intervallo[1]]), label='trigger')
        plt.xlim(min(xvec[intervallo[0]:intervallo[1]]),max(xvec[intervallo[0]:intervallo[1]]))
        #plt.legend(loc='upper left')
        #plt.show()
        plt.savefig(savedir+str(int(timestamps[j]))+'_triplot_resonant')
        plt.close()

def savetimestamps(dir, vec):
    vec=np.array(vec)
    np.savetxt(dir+'Timestamps.txt', vec.astype(int), fmt='%i')

def terzitrigger(segnale, sampling, centerband):
    trigband=[]
    sampling=44100
    #a=np.arange(-19,19)
    #centerbands=1000*(2**(a/3))
    #for j in range (0,len(centerbands)):
    trigband=(markholes(ac.octavepass(prova[1],centerband, fs=sampling, order=8, fraction=3), sampling, centerband, 0))
    return (trigband)

#Crea una finestra di hann, perché a tutti piace averne una sempre a disposizione
s=0.01
N=math.floor(44100*s)
wind=signal.windows.hann(N)

#runbelle è un vettore di intervalli (inizio, fine) scritto in secondi di asfalto bello, per ora carica solo i dati di coltano e non accetta parametri in input, ma magari un giorno può diventare qualcosa
def LeqSezioni():
    runbelle=np.loadtxt(loadir+'runbelle_coltano.txt')
    runmedie=np.loadtxt(loadir+'runmedie_coltano.txt')
    runbrutte=np.loadtxt(loadir+'runbrutte_coltano.txt')
    a=[]
    for i in range (0,len(runbelle)):
        print (i)
        a.append(calcoleq(prova[1], runbelle[i]))
    a=np.array(a)
    b=[]
    for i in range (0,len(runmedie)):
        print (i)
        b.append(calcoleq(prova[1], runmedie[i]))
    b=np.array(b)
    c=[]
    for i in range (0,len(runbrutte)):
        print (i)
        c.append(calcoleq(prova[1], runbrutte[i]))
    c=np.array(c)
    return(a,b,c)
    
def Yuleintervallo(segnale, intervallo, deltat):
    secondi=intervallo[0]
    while (secondi < (intervallo[1])):
        intervallino=((secondi+deltat-0.1),(secondi+deltat+0.1))
        plottayule(segnale,(intervallino[0],intervallino[1]))
        plt.ylim(-65,15)
        plt.savefig(savedir+str(int(secondi*100))+'_YW_psd')
        plt.close()
        secondi+=0.04

def calcolav(tratto, secondo):
    tratto=butter_lowpass_filter(tratto[campioni(secondo-0.2):campioni(secondo+0.2)],20,44100)
    #tratto=rollingavgpd(tratto,5000)
    massimi=scipy.signal.find_peaks(tratto)
    #max = stats.trim_mean(massimi[0], 0.1) # Trim 10% at both ends
    #Tmedio=stats.trim_mean(massimi[0][1:]-massimi[0][:-1], 0.3)/44100
    Tmediano=np.median(massimi[0][1:]-massimi[0][:-1])/44100
    vmedia=(np.pi*0.656)/Tmediano
    #adesso istogrammo ogni intervallo di un secondo, e se c'è un massimo restituisco quello, altrimenti la media tra massimi (bisogna pur vivere)
    #histo=np.histogram(massimi[0][1:]-massimi[0][:-1])
    #Treal=np.average(histo[1][np.where(histo[0]==np.max(histo[0]))])/44100
    # vmedia=(np.pi*0.656)/Treal
    return(vmedia)
    
def calcolavmin(tratto, secondo):
    tratto=butter_lowpass_filter(tratto[campioni(secondo-0.5):campioni(secondo+0.5)],40,44100)
    #tratto=rollingavgpd(tratto,5000)
    massimi=scipy.signal.find_peaks(-(tratto[tratto**2 < 2000])**2)
    massimitrim=scipy.stats.trimboth(massimi[0], 0.2)
    # Tmedio=stats.trim_mean(massimi[0][1:]-massimi[0][:-1], 0.2)/44100
    Tmedio=np.median(massimitrim[1:]-massimitrim[:-1])/44100
    Tmax=Tmedio+0.5*np.std(massimitrim[1:]-massimitrim[:-1])/44100
    Tmin=Tmedio-0.5*np.std(massimitrim[1:]-massimitrim[:-1])/44100
    vmedia=(np.pi*0.656)/(2*Tmedio)
    vmax=(np.pi*0.656)/(2*Tmin)
    vmin=(np.pi*0.656)/(2*Tmax)
    return(kph(vmedia), kph(vmin), kph(vmax))

def genvecv(strada):
    s=[]
    for j in range (1, math.floor(secondi(len(strada)))):
       s.append(calcolavmin(strada, j))
    return (np.array(s)) 
       
def distribuzionepicchi(tratto, secondo):
    tratto=butter_lowpass_filter(tratto[campioni(secondo-10):campioni(secondo+10)],20,44100)
    massimi=scipy.signal.find_peaks(tratto)
    a=(massimi[0][1:]-massimi[0][:-1])/44100
    v=(np.pi*0.656)/a
    return v
    

def distribuzioneminimi(tratto, secondo):
    tratto=butter_lowpass_filter(tratto[campioni(secondo-1):campioni(secondo+1)],20,44100)
    massimi=scipy.signal.find_peaks(-(tratto[tratto**2 < 1000])**2)
    a=(massimi[0][1:]-massimi[0][:-1])/44100
    v=(np.pi*0.656)/a
    return v
    
def picchiyule (data, secondo):
    a=plottayule(data, (secondo-0.5, secondo+0.5))
    psd=a.psd
    spaziatura=(np.array(a.frequencies()[1:])-np.array(a.frequencies()[:-1]))[-1]
    massimi=scipy.signal.find_peaks(psd)[0]*spaziatura
    massimires=massimi[massimi>150]
    return (massimires[:3])

#Calcola e restituisce il dataset per il machine learning nel pacchetto data mandato in input (usa almeno .4 secondi)
def calcolafeaturesmasino(data, tipo):
    Power=poweryule(data)
    Power5k=powerbandyw(data, 5000,22000)
    Ratio5k=Power5k/Power
    Ratio1res=poweratio(data, 366,495,100,5000)
    Primotoro=powerbandyw(data,175,245)
    a=np.array((Power,Power5k,Ratio5k,Ratio1res,Primotoro,tipo))
    return(a)


def calcolafeatimestamp (data, ts, delta):
    ts=ts-delta
    a=[]
    for j in range (0, len(ts)):
        print (ts[j])
        feat1=calcolafeaturesipool(data[campioni(ts[j]-0.2): campioni(ts[j])], str(ts[j]+delta))
        feat2=calcolafeaturesipool(data[campioni(ts[j]-0.1): campioni(ts[j]+0.1)], str(ts[j]+delta))
        feat3=calcolafeaturesipool(data[campioni(ts[j]): campioni(ts[j]+0.2)], str(ts[j]+delta))
        feat=np.concatenate((feat1[:-1],feat2[:-1],feat3))
        a.append(feat)
    adf=pd.DataFrame(a, index=np.arange(len(a)), columns=('Total_power1', 'Firstres1', 'Ratio_1res1', 'Ratio_2res1', 'Ratio_3res1', 'Ratio_hifreq1', 'Total_power2', 'Firstres2', 'Ratio_1res2', 'Ratio_2res2', 'Ratio_3res2', 'Ratio_hifreq2', 'Total_power3', 'Firstres3', 'Ratio_1res3', 'Ratio_2res3', 'Ratio_3res3', 'Ratio_hifreq3', 'Label'))
    return adf




def calcolafeaturesipool(data, tipo):
    a=getyule(data, (0,secondi(len(data))))
    psd=a.psd
    Psd=psd
    #vettore delle frequenze
    frequenze=np.array(a.frequencies())
    spaziatura=(frequenze[1:]-frequenze[:-1])[-1]
    #comincio a tagliare in frequenza
    integrale=np.sum(spaziatura*psd)
    Power=2*integrale

    freq1=200
    freq2=240
    #comincio a tagliare in frequenza
    frequenze=np.array(a.frequencies())
    cutbasso=frequenze[frequenze>freq1]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*psd[indicebasso:indicealto])
    Primotoro=2*integrale
    
    freq3=270
    freq4=400
    frequenze=np.array(a.frequencies())
    cutbasso=frequenze[frequenze>freq1]
    cutbasso1=frequenze[frequenze>freq3]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    indicebasso1=np.where(frequenze>cutbasso1[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    band1=cutbasso1[cutbasso1<freq4]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    indicealto1=(np.where(frequenze<band1[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*Psd[indicebasso:indicealto])
    integrale1=np.sum(spaziatura*Psd[indicebasso1:indicealto1])
    Ratioprimo= integrale/integrale1
    
    freq1=400
    freq2=460
    freq3=480
    freq4=620
    frequenze=np.array(a.frequencies())
    cutbasso=frequenze[frequenze>freq1]
    cutbasso1=frequenze[frequenze>freq3]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    indicebasso1=np.where(frequenze>cutbasso1[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    band1=cutbasso1[cutbasso1<freq4]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    indicealto1=(np.where(frequenze<band1[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*Psd[indicebasso:indicealto])
    integrale1=np.sum(spaziatura*Psd[indicebasso1:indicealto1])
    Ratiosecondo= integrale/integrale1

    freq1=630
    freq2=660
    freq3=700
    freq4=830
    frequenze=np.array(a.frequencies())
    cutbasso=frequenze[frequenze>freq1]
    cutbasso1=frequenze[frequenze>freq3]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    indicebasso1=np.where(frequenze>cutbasso1[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    band1=cutbasso1[cutbasso1<freq4]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    indicealto1=(np.where(frequenze<band1[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*Psd[indicebasso:indicealto])
    integrale1=np.sum(spaziatura*Psd[indicebasso1:indicealto1])
    Ratioterzo= integrale/integrale1
    
    freq1=2500
    freq2=8000
    freq3=1500
    freq4=2000
    frequenze=np.array(a.frequencies())
    cutbasso=frequenze[frequenze>freq1]
    cutbasso1=frequenze[frequenze>freq3]
    indicebasso=np.where(frequenze>cutbasso[0])[0][0] #indice della frequenza più bassa
    indicebasso1=np.where(frequenze>cutbasso1[0])[0][0] #indice della frequenza più bassa
    band=cutbasso[cutbasso<freq2]
    band1=cutbasso1[cutbasso1<freq4]
    indicealto=(np.where(frequenze<band[-1]))[0][-1] #indice della frequenza più alta
    indicealto1=(np.where(frequenze<band1[-1]))[0][-1] #indice della frequenza più alta
    integrale=np.sum(spaziatura*Psd[indicebasso:indicealto])
    integrale1=np.sum(spaziatura*Psd[indicebasso1:indicealto1])
    Ratiohifreq= integrale/integrale1
    
    
    # Power=poweryule(data)
    # Primotoro=powerbandyw(data, 200,240)
    # Ratioprimo=poweratio(data, 200,240, 270,400)
    # Ratiosecondo=poweratio(data, 400,460,480,620)
    # Ratioterzo=poweratio(data,630,660,700, 830)
    # Ratiohifreq=poweratio(data,2500,8000,1500, 2000)
    a=np.array((Power,Primotoro,Ratioprimo,Ratiosecondo,Ratioterzo,Ratiohifreq,tipo))
    return(a)

def campionadati(data, rangebello, rangebrutto, nsamples):
    feat=[]
    for j in range (0, nsamples):
        rand=np.random.random()
        secbello=(0.4+rangebello[0]+(rangebello[1]-rangebello[0])*rand)
        secbrutto=(0.4+rangebrutto[0]+(rangebrutto[1]-rangebrutto[0])*rand)
        featbello=calcolafeaturesipool(data[campioni(secbello-0.4): campioni(secbello)], 'bello')
        featbrutto=calcolafeaturesipool(data[campioni(secbrutto-0.4): campioni(secbrutto)], 'brutto')
        feat.append(featbello)
        feat.append(featbrutto)
    featdf=pd.DataFrame(feat, index=np.arange(len(feat)), columns=('Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq', 'Label'))
    #featdf=featdf.dropna(how='all')
    featdf[['Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq']] = featdf[['Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq']].apply(pd.to_numeric)
    return (featdf)

def campionadativ2(data, rangebello, rangebrutto, nsamples):
    feat=[]
    for j in range (0, nsamples):
        rand=np.random.random()
        secbello=(0.4+rangebello[0]+(rangebello[1]-rangebello[0])*rand)
        secbrutto=(0.4+rangebrutto[0]+(rangebrutto[1]-rangebrutto[0])*rand)
        featbello1=calcolafeaturesipool(data[campioni(secbello-0.4): campioni(secbello-0.2)], 'bello')
        featbrutto1=calcolafeaturesipool(data[campioni(secbrutto-0.4): campioni(secbrutto-0.2)], 'brutto')
        featbello2=calcolafeaturesipool(data[campioni(secbello-0.3): campioni(secbello-0.1)], 'bello')
        featbrutto2=calcolafeaturesipool(data[campioni(secbrutto-0.3): campioni(secbrutto-0.1)], 'brutto')
        featbello3=calcolafeaturesipool(data[campioni(secbello-0.2): campioni(secbello)], 'bello')
        featbrutto3=calcolafeaturesipool(data[campioni(secbrutto-0.2): campioni(secbrutto)], 'brutto')
        featbello=np.concatenate((featbello1[:-1],featbello2[:-1],featbello3))
        featbrutto=np.concatenate((featbrutto1[:-1],featbrutto2[:-1],featbrutto3))
        feat.append(featbello)
        feat.append(featbrutto)
        featdf=pd.DataFrame(feat, index=np.arange(len(feat)), columns=('Total_power1', 'Firstres1', 'Ratio_1res1', 'Ratio_2res1', 'Ratio_3res1', 'Ratio_hifreq1', 'Total_power2', 'Firstres2', 'Ratio_1res2', 'Ratio_2res2', 'Ratio_3res2', 'Ratio_hifreq2', 'Total_power3', 'Firstres3', 'Ratio_1res3', 'Ratio_2res3', 'Ratio_3res3', 'Ratio_hifreq3', 'Label'))
    #featdf=featdf.dropna(how='all')
    featdf[['Total_power1', 'Firstres1', 'Ratio_1res1', 'Ratio_2res1', 'Ratio_3res1', 'Ratio_hifreq1', 'Total_power2', 'Firstres2', 'Ratio_1res2', 'Ratio_2res2', 'Ratio_3res2', 'Ratio_hifreq2', 'Total_power3', 'Firstres3', 'Ratio_1res3', 'Ratio_2res3', 'Ratio_3res3', 'Ratio_hifreq3']] = featdf[['Total_power1', 'Firstres1', 'Ratio_1res1', 'Ratio_2res1', 'Ratio_3res1', 'Ratio_hifreq1', 'Total_power2', 'Firstres2', 'Ratio_1res2', 'Ratio_2res2', 'Ratio_3res2', 'Ratio_hifreq2', 'Total_power3', 'Firstres3', 'Ratio_1res3', 'Ratio_2res3', 'Ratio_3res3', 'Ratio_hifreq3']].apply(pd.to_numeric)
    return (featdf)        
        
#lancia calcolafeatures su un intervallo largo dividendolo in pezzi della stessa larghezza, specificata in input in secondi (intervalli). Returna un array di array, vuole la classificazione della pavimentazione già fatta dall'utente. Ricorda che deve essere una stringa.
def arrayfeatures(data, intervalli, classificazione):
    feat=[]
    j=intervalli
    while j<secondi(len(data)):
        feat.append(calcolafeaturesipool(data[campioni(j-intervalli): campioni(j)], classificazione))
        j+=intervalli
    plt.close()
    return np.array(feat)

#costruisce un dataframe di pandas partendo da due array numpy returnati da arrayfeatures per accorparli in un unico dataset
def wrapdf (data1,data2):
    df1 = pd.DataFrame(data1, index=np.arange(len(data1)), columns=('Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq', 'Label'))
    df2 = pd.DataFrame(data2, index=np.arange(len(data1), len(data2)+len(data1)),columns=('Total_power', 'Firstres','Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq', 'Label'))
    df=pd.concat([df1,df2])
    df[['Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq']] = df[['Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq']].apply(pd.to_numeric)
    return(df)

# Fa lo stesso di wrapdf ma usando dataframe di pandas
def wrapandas (df1,df2):
    df2.set_index(np.arange(len(df1), len(df1)+len(df2))) 
    df=pd.concat([df1,df2])
    df.set_index(np.arange(len(df1)+len(df2)))
    df[['Total_power', 'Power5k', 'Ratio5k', 'Ratio_2res', 'Power_firstres']] = df[['Total_power', 'Power5k', 'Ratio5k', 'Ratio_2res', 'Power_firstres']].apply(pd.to_numeric)
    return(df)

# bello=arrayfeatures(prova[1][campioni(50):campioni(70)], 0.4, 'bello')
# brutto=arrayfeatures(prova[1][campioni(95):campioni(115)], 0.4, 'brutto')
# dataframe=wrapdf(bello, brutto)

# b=[]
# for j in filelistph:
#     a=loadwav(j)
#     if secondi(len(a[1]))>0.4:
#         datanp=arrayfeatures(a[1][0:campioni(math.floor(secondi(len(a[1]))/0.4)*0.4)], 0.4, 'ph')
#         df=pd.DataFrame(datanp, index=np.arange(len(datanp)), columns=('Total_power', 'Firstres', 'Ratio_1res', 'Ratio_2res', 'Ratio_3res', 'Ratio_hifreq', 'Label'))
#         b.append(df)
    
def nearest(dataset):
    X=dataset.drop(columns=['Label','Firstres','Ratio_1res','Ratio_3res','Ratio_2res'])
    print (X)
    y=dataset['Label'].values
    X_train, X_test, y_train, y_test=train_test_split(X,y, stratify=y, test_size=0.7, random_state=42)
    nca = NeighborhoodComponentsAnalysis(random_state=87)
    knn = KNeighborsClassifier(n_neighbors=3)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)
    print(nca_pipe.score(X_test, y_test))
    return (nca_pipe)
    
def psdogramma(segnale, secondo):
    a=[]
    step=0.1
    length=2000
    IRF=plottapsdbucata(segnale, (0, len(segnale)))
    plt.close()
    for j in range (0,length):
        tempo=secondo+(j-math.floor(length/2))*step
        print (tempo)
        psd=plottapsdbucata(prova[1],(tempo,tempo+step))
        a.append(np.log10(psd[0][2:64+2]/IRF[0][2:64+2]))
        plt.close()
    return (a, psd[1][2:64+2])
    

# # Crea un vettore che campiona le frequenze delle prime risonanze 
# primeres=[]
# for j in range (1,101):
#     a=plottayule(prova[1],(10+(j-1)*0.4, 10+j*0.4))
#     picco=sp.signal.find_peaks(a.psd)[0]
#     primeres.append(picco[picco>160/2.7][0]*2.691650390625)
#     plt.close()
#     print (j)
    

 
# # define bounds of the domain
# min1, max1 = X['Total_power'].min()-1, X['Total_power'].max()+1
# min2, max2 = X['Ratio_hifreq'].min()-1, X['Ratio_hifreq'].max()+1
# #uniform range across the dmain
# x1grid = arange(min1, max1, (max1-min1)/500)
# x2grid = arange(min2, max2, (max2-min2)/500)
# # create all of the lines and rows of the grid
# xx, yy = np.meshgrid(x1grid, x2grid)
# # flatten each grid to a vector
# r1, r2 = xx.flatten(), yy.flatten()
# r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# # horizontal stack vectors to create x1,x2 input for the model
# grid =np.hstack((r1,r2))
# # make predictions for the grid
# yhat = classificatore.predict(grid)
# # reshape the predictions back into a grid
# zz = yhat.reshape(xx.shape)
# ZZ=zz=='brutto'
# # plot the grid of x, y and z values as a surface
# plt.contourf(xx, yy, ZZ,cmap='Paired')
# #plt.contour(xx, yy, ZZ,cmap='Paired')
# sns.scatterplot(x="Total_power", y="Ratio_hifreq", hue='Label', data=dataset)
# #plt.xlim(min(xx[0]), max(xx[0]))
# #plt.ylim(min(yy[0]), max(yy[-1]))
# plt.show()
#     
#dataset.to_csv(r'F:/Misure navacchio 2/pint/unzipped/run_spezzate/dataset.csv')


# # Violin plot
#  sns.violinplot(y='Label',x='Total_power', data=todo, inner='quartile')
#  sns.pairplot(testset, hue='Label', markers='+')
    
    
# # Confusion matrix plot: prima createla poi plottala col seguente
# group_names = ['True Neg','False Pos','False Neg','True Pos']
# #group_counts = [“{0:0.0f}”.format(value) for value in
#                 #cf_matrix.flatten()]
# #group_percentages = ["{0:.2%}".format(value) for value in
#                      #cf_matrix.flatten()/np.sum(cf_matrix)]
# group_percentages= ["{0:.2%}".format(cf_matrix[0][0]/np.sum(cf_matrix[0])),"{0:.2%}".format(cf_matrix[0][1]/np.sum(cf_matrix[0])),  "{0:.2%}".format(cf_matrix[1][0]/np.sum(cf_matrix[1])),"{0:.2%}".format(cf_matrix[1][1]/np.sum(cf_matrix[1]))]
# labels = [f"{v1}\n{v2}" for v1, v2 in
#           zip(group_names,group_percentages)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# # crea una confusion matrix
# actual=np.concatenate((np.zeros(86)+1,np.zeros(153-86)))
# predicted=np.concatenate((np.zeros(2),np.zeros(151)+1))
# matrix = confusion_matrix(actual,predicted, labels=[1,0])
# tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
# print('Outcome values : \n', tp, fn, fp, tn)
# df_cm = pd.DataFrame(matrix, range(2), range(2))
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
# plt.show()
# 
# matrix = classification_report(actual,predicted,labels=[1,0])
# print('Classification report : \n',matrix)
# Plot overload example:
# for j in range (0,len(runbelle)):
#     plottaspettrosbucato(prova[1],(runbelle[j][0],runbelle[j][1]))

#O gaussiana se vuoi
#wind=signal.windows.gaussian(N,round(44100/200))

# plt.plot(np.linspace(0,secondi(len(tagliato)), num=len(tagliato)),PtoLeq(tagliato))
# plt.show()
# dove=np.where(z>4)
# 
# for j in range (0,len(dove[0])):
#     trigUP[dove[0][j]:dove[0][j]+longroll]=1
    
# Triggah=triggernew(prova[1], 10000, 200)
# s=0.01 #(durata in secondi della finestra mobile)
# #creo la finestra' + str(j+1) + '' + str(j+1) + '
# #wind=signal.windows.hann(N)
# 
# plt.subplot(313)
# plt.plot(Triggah[0], Triggah[1])
# 
# plt.subplot(312)
# plottaserietemporale(np.sqrt(prova[1]**2),prova[0])
