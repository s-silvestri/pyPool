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
import scipy as sp
#detection centrata nella buca con mezzo secondo di dati
#dovrai usare pywt.cwt()
windows=False

@plt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%2d}$' % (x)
    
def secondi(campioni):
    return (campioni/44100)
    
def campioni(secondi):
    return (math.floor(secondi*44100))

def PtoLeq(P):
    P0=2e-5
    return (10*np.log10(P**2/P0**2))
    
def varianzastupida(set, length):
    var=[]
    for j in range (0, len(set)-length):
        var.append(np.var(set[j:j+length]))
    return(var)
    
def varianzaintelligente(set, length):
    var=[]
    media=[]
    epsilon=[]
    for j in range (0,length):
        var.append(np.var(set[0:j]))
        media.append(np.mean(set[0:j]))
        epsilon.append(np.abs(np.sum(set[0:j]-media[j])))
    for j in range (length+1, len(set)-length):
        media.append((set[j]-set[j-length])/length)
        epsilon.append(np.abs(np.sum(set[j-length:j]-media[-1])))
        var.append(1)
    return(epsilon)

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

def rollingtrig(serie, lungo, corto):
    seriepd=pd.Series(serie)
    largemean=seriepd.rolling(lungo).median()
    largevar=seriepd.rolling(lungo).var()
    smallmean=seriepd.rolling(corto).median()
    return (smallmean.to_numpy(), largemean.to_numpy(), largevar.to_numpy())

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

#Fa la trasformata di wavelet suddividendo gli intervalli di frequenze in nlogbin, da migliorare. Notare che le etichette sulle y fanno schifo, sono 4 punti presi all'interno dell'intervallo e approssimati all'esponenziale più vicino, quindi se vedi 10^3 può essere taggato anche in 2.35*10^3. Modi più intelligenti per fare questa cosa includono lo scegliere i bin logaritmici in modo che vengano particolarmente in accordo con 3 divisioni della scala o scrivere l'asse y in modo non del tutto automatico.
def wavelet (sig, nlogbin):
    fsampling=44100
    #logbins=np.logspace(1,4,num=nlogbin)
    #dt=0.01
    #frequencies = pywt.scale2frequency('cmor3-1',np.arange(0,nlogbin)) / 0.1
    #widths=np.logspace(0,4,num=nlogbin)
    widths=np.logspace(np.log10(fsampling/5000),np.log10(fsampling/10), num=nlogbin)
    cwtmatr, freqs = pywt.cwt(sig, widths, 'cmorl3-1')
    #logextent=[np.log(4.41),np.log10(4410)]#[np.log10((freqs[-1]+freqs[-2])*44100/2),np.log10((freqs[1]+freqs[0])*44100/2)]
    extent=[(np.min(1/(widths/44100))), (np.max(1/(widths/44100)))]
    logextent=np.log10(extent)
    fig, (ax) = plt.subplots()
    ax.yaxis.set_major_locator(plt.MaxNLocator(len(widths)))
    #ax.yaxis.set_major_formatter(fake_log)
    imm=ax.imshow(np.abs(cwtmatr), aspect='auto', cmap='jet', vmax=abs(cwtmatr).max(), vmin=abs(cwtmatr).min(), extent=[0,len(sig)/fsampling, logextent[0], logextent[1]]) 
    ax.set_ylabel('Log frequency')
    ax.set_xlabel('time [s]')
    fig.colorbar(imm)

def trigger(segnale, largewindow, smallwindow):
    sampling=44100
    rollingsmall=(rollingavg(np.sqrt(segnale**2), 200)) #[(largewindow-smallwindow):])
    rollinglarge=(rollingavgnonzero(np.sqrt(segnale**2), 10000, rollingsmall))
    nsecutili=math.floor(len(rollingsmall)/sampling)#[:-(largewindow-smallwindow)])/sampling)
    rollinglargeaffettato=rollinglarge[:nsecutili*sampling]
    rollingsmallaffettato=rollingsmall[:nsecutili*sampling]
    trigger=np.divide(rollingsmallaffettato,rollinglargeaffettato)
    xtime=np.linspace(0, len(rollinglargeaffettato)/44100, num=len(rollinglargeaffettato))
    return(xtime, trigger, rollingsmallaffettato, rollinglargeaffettato)
    


loadir='/home/kibuzo/Rotoliamo/Dati/misura/'
loadir='/home/kibuzo/Rotoliamo/Topate/tempdata/'
loadir='/media/kibuzo/Gatto Silvestro/Buche/Misure Coltano/pint/unzipped/run_spezzate/'


if windows:
    loadir='C:/Users/acust/Desktop/misura/Longnavacchio/'
    savedir='c:/Users/acust/Desktop/misura/processed'

filelist=[]
for filename in os.listdir(loadir):
    if filename.endswith(".wav"):
        filelist.append(loadir+filename)

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

# plt.plot(np.linspace(0,secondi(len(tagliato)), num=len(tagliato)),PtoLeq(tagliato))
# plt.show()
# dove=np.where(z>4)
# 
# for j in range (0,len(dove[0])):
#     trigUP[dove[0][j]:dove[0][j]+longroll]=1
    
savedir='/media/kibuzo/Gatto Silvestro/Buche/Misure navacchio/gopro/00082/triggered/plot/'
savedir='/media/kibuzo/Gatto Silvestro/Buche/Misure Coltano/pint/unzipped/run_spezzate/Triggered/2khz/plot/'
prova=loadwav(filelist[-1])
# Triggah=triggernew(prova[1], 10000, 200)
# s=0.01 #(durata in secondi della finestra mobile)
# #creo la finestra
# N=math.floor(44100*s)
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