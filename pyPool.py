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
import scipy as sp
import scipy.fftpack
from spectrum import *
windows=False
#loadwav carica file e restituisce un vettore dove il primo elemento è la frequenza di sampling, il secondo è la serie numerica. Il numero di conversione serve per passare da canali adc a pressione (canali adc 32767 e pressione in pascal 813.5330)
def loadwav(file):
    samplerate, data = read(file)
    return (samplerate, data*813.5330/32767) # quel numerino è per la calibrazione del sensore audio

#---------------------- Qui ci sono un po' di indirizzi di cartelle da cui volendo si caricano facilmente i file: ognuno le aggiorni con le sue folder preferite, idealmente tutti i file sono in una cartella poi vengono caricate in un vettore filelist[].
loadir='/media/kibuzo/Gatto Silvestro/Buche/Misure Coltano/pint/unzipped/run_spezzate/'

savedir='/media/kibuzo/Gatto Silvestro/Buche/Misure navacchio/gopro/00082/triggered/plot/'
savedir='/media/kibuzo/Gatto Silvestro/Buche/Misure Coltano/pint/unzipped/run_spezzate/Triggered/2khz/plot/'


if windows:
    loadir='C:/Users/acust/Desktop/misura/Long/'
    savedir='c:/Users/acust/Desktop/misura/processed/'

#Questo mette tutti i nomi dei file in un apposito vettore.
filelist=[]
for filename in os.listdir(loadir):
    if filename.endswith(".wav"):
        filelist.append(loadir+filename)
prova=loadwav(filelist[0])

#---------------------- Cominciano le funzioni

#Copiato da internet. Usato, e poi abbandonato, per l'asse verticale logaritmico della wavelet: ha diversi problemi, tipo che ti approssima 5x10^23 in 10^23, magari un giorno lo aggiusto
@plt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%2d}$' % (x)

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
    specgramma=plt.specgram(serie,Fs=sampling, window=wind, scale='dB', NFFT=N, cmap='jet')
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
    plt.psd(data, NFFT=math.floor(len(data)/16), Fs=44100, detrend=None)
    plt.xscale('log')
    #ax.semilogx(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlim(100,12800)
    plt.xlabel('Frequency[Hz]')
    plt.show()
    
def plottarmapsdsbucata(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    data=sbucastrada(data,(intervallo[0],intervallo[1]))[1]
    N = len(data)
    p = parma(data, 120, 120, 240, NFFT=len(data), sampling=44100)
    #psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
    p()
    p.plot(sides='centerdc')
    plt.xscale('log')
    plt.xlim(100,15000)
    plt.show()

def plottayule(data,intervallo):
    sampling=44100
    intervallo=np.array(intervallo)
    data=sbucastrada(data,(intervallo[0],intervallo[1]))[1]
    N = len(data)
    p = pyule(data, 500, NFFT=len(data), sampling=44100)
    #psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
    p()
    p.plot(sides='centerdc')
    plt.xscale('log')
    plt.xlim(100,12800)
    #plt.show()
    

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
    longroll=math.floor((sampling*25)/centerband)
    shortroll=math.floor(sampling/(2*centerband))
    trig=rollingtrig(np.sqrt(segnale**2), longroll, shortroll)
    #varianza=np.array(trig[2])
    #varianzaback=np.concatenate((varianza[longroll:],np.zeros(longroll)))
    z=np.sqrt((trig[0]-trig[1])**2/trig[2])
    trigUP=z>4
    #varianza=varianza-(varianza-varianzaback)*trigUP
    timestamp=[]
    x=[]
    mu=[]
    sigma=[]
    j=0
    while (j< len(trig[1])):
        if (z[j])>4:
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
        intervallo=[math.floor((timestamps[j]+deltat-1)*44100),math.floor((timestamps[j]+deltat+1)*44100)]
        wavelet(segnale[intervallo[0]:intervallo[1]],20)
        plt.savefig(savedir+str(int(timestamps[j]))+'_cwt')
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
