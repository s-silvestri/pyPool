import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import signal
import math
from scipy.io import wavfile
import wavio
from scipy import interpolate
import soundfile as sf
from librosa import resample
import scipy.signal as sps
#Todo: Una CDF diversa per ogni banda in frequenza per i transienti. Però andrebbero identificati come righe, non come intervalli, se sono crack. La strada brutta, on the other hand, non è un crack ma è un brutto uniforme, e ci si aspetta che porti a un aumento del livello medio broadband.
file='F:/pint/Misure Coltano/pint/unzipped/run_spezzate/giro_lungo.wav'
sampling_rate, data=wavfile.read(file)
durabin=0.2 #Durata in secondi di ogni intervallo su cui fare la PSD
#Data_IRF=data[559*44100:594*44100]
IRF=plottapsdbucata(data, (559, 594)) #Asse X è in IRF[1], asse Y è in IRF[0], moltiplica sempre il segnale per IRF[0] e plotta irf[1],sig*irf[0]
IRF=np.array(IRF)
#IRF[0]/=(35/durabin) #Riporto la potenza a valori contenuti in 0.2 secondi
plt.close() #per rivederlo: plt.loglog(IRF[1], IRF[0])
pezzobello=data[559*44100:594*44100]
crack=data[167*44100:174*44100]
scale=813.5330/32767


#Da debuggare, vedi descrizione.
def PSDToSeries(lenTimeSeries,freq,psdLoad):
    '''
    Genera una serie temporale compatibile con la PSD fornita in input. Per qualche ragione mi raddoppia le frequenze. Vabè.

    '''
    #
    #Intervallo in frequenza
    df=freq[1]-freq[0]
    #print('df = ', df)

    spettro0=psdLoad
    irf0=IRF[1]
    f=interpolate.interp1d(irf0, spettro0, kind='linear')
    irf1=np.linspace(IRF[1][0], IRF[1][-1], num=int(lenTimeSeries))
    spettro1=f(irf1)
    binwidth=(irf1[1:]-irf1[:-1])[0]
    #Ampiezze spettrali, prese dai bin PSD: attenzione che è dilatato dall'oversampling
    amplitude=np.sqrt(4*spettro1*binwidth)

    #creo i vettori
    epsilon=np.zeros((len(amplitude)))
    randomSeries=np.zeros((len(amplitude)))


    #Creo la serie temporale: fase random tra -pi e pi
    #Generate random phases between [-2pi,2pi]
    epsilon=np.pi * (2*np.random.randn(1,len(amplitude))-1)

    #Inverse Fourier
    randomSeries=np.real(np.fft.ifft(amplitude*np.exp(epsilon*1j*2*np.pi)))

    return np.reshape(randomSeries, np.shape(randomSeries)[1])#faglielo direttamente riuscire col resampling perché qui è a 22050 Hz

# Genera un rumore bianco di media mean varianza std lungo seconds (campionato a 44100 Hz)
def genwhite (mean, std, seconds):
    num_samples = 44100
    samples = np.random.normal(mean, std, size=math.floor(seconds*num_samples))
    return (samples)

def genstradabella(s):
    return(genwhite(1,5.54e-3,s))

# def genspettrobello(irf=IRF):
#     logmean=1.0186691440524822
#     logvar=0.1763178502675603
#     mean=-2*log(logmean)-0.5*np.log(logvar+logmean**2)
#     std=np.sqrt(-2*log(logmean)+np.log(logvar+logmean**2))
#     #std=0.42
#     spettro=np.random.lognormal(mean, std, 2049)
    return (spettro*IRF[0])

def genspettroIRF(CDF, pdf):
    spettro=FromCDF(CDF,pdf)
    return (np.array(spettro)*IRF[0])


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

def FromCDF (CDF, pdf):
    CDF=np.array(CDF)
    randx=[]
    for j in range (0,2049):
        xi=np.random.random()
        randx.append(pdf[1][np.max(np.where(CDF<xi))])
    return(randx)

#Inferenza: si crea una psd in un punto a caso di un intervallo bello di durata 0.2 secondi, divide per la IRF e calcola il valor medio. La sigma delle medie sarà la varianza campionaria da mertere in genstradabella. Vado da 0:93 per fermarmi a 1kHz
avg=[]
psdarray=[]
for j in range (0, 1000):
    rand=np.random.random()*33
    psdbello=plottapsdsbucata(pezzobello, (rand, rand+durabin))
    psdarray.append(psdbello[0])
    plt.close()
    #avg.append(np.mean(psdbello[0]/IRF[0])) # Falsava i risultati lisciando con la media
    avg=avg+((psdbello[0]/IRF[0]).tolist())
pdf=plt.hist(avg, bins=10000)
plt.close()
cdf=[]
for j in range (0,len(pdf[0])):
    cdf.append(np.sum(pdf[0][0:j]/np.sum(pdf[0])))

avgcrack=[]
psdarraycrack=[]
for j in range (0, 1000):
    rand=np.random.random()*6
    psdcrack=plottapsdsbucata(crack, (rand, rand+durabin))
    psdarraycrack.append(psdcrack[0])
    plt.close()
    #avg.append(np.mean(psdbello[0]/IRF[0])) # Falsava i risultati lisciando con la media
    avgcrack=avg+((psdcrack[0]/IRF[0]).tolist())
pdfcrack=plt.hist(avgcrack, bins=10000)
plt.close()
cdfcrack=[]
for j in range (0,len(pdf[0])):
    cdfcrack.append(np.sum(pdfcrack[0][0:j]/np.sum(pdfcrack[0])))


psdarray=np.array(psdarray)
avgarray=[]
stdarray=[]
for j in range (0, len(psdarray[1])):
    avgarray.append(np.mean(psdarray[:,j]/IRF[0][j]))
    stdarray.append(np.std(psdarray[:,j]/IRF[0][j]))


#Crossscheck: faccio lo stesso con lo spettro sintetico
avgsynth=[]
avgsyntharray=[]
psdarrasynth=[]
for j in range  (0,1000):
    psdarrasynth.append(genspettroIRF(cdf,pdf))
    avgsynth.append(np.mean(psdarrasynth[j][0:93]/IRF[0][0:93]))
psdarrasynth=np.array(psdarrasynth)
avgarraysynth=[]
stdarraysynth=[]
primoquartile=[]
terzoquartile=[]
for j in range (0, len(psdarrasynth[1])):
    avgarraysynth.append(np.mean(psdarrasynth[:,j]/IRF[0][j]))
    primoquartile.append(np.percentile(psdarrasynth[:,j]/IRF[0][j],25))
    terzoquartile.append(np.percentile(psdarrasynth[:,j]/IRF[0][j],75))
    stdarraysynth.append(np.std(psdarrasynth[:,j]/IRF[0][j]))

def plottaquartilisintetici():
    plt.loglog(IRF[1], avgarraysynth*IRF[0], label='median psd')
    plt.loglog(IRF[1], primoquartile*IRF[0], label='first quartile', linestyle='--')
    plt.loglog(IRF[1], terzoquartile*IRF[0], label='third quartile', linestyle='--')
    plt.ylim(1e-4,3e3)
    plt.legend()

def plottaquartiliveri(PSD=psdarray):
    avgarrayvero=[]
    primoquartile=[]
    terzoquartile=[]
    for j in range (0,np.shape(PSD)[1]):
        avgarrayvero.append(np.mean(PSD[:,j]/IRF[0][j]))
        primoquartile.append(np.percentile(PSD[:,j]/IRF[0][j],25))
        terzoquartile.append(np.percentile(PSD[:,j]/IRF[0][j],75))
    plt.loglog(IRF[1], avgarrayvero, label='median psd')
    plt.loglog(IRF[1], primoquartile, label='first quartile', linestyle='--')
    plt.loglog(IRF[1], terzoquartile, label='third quartile', linestyle='--')
    plt.ylim(1e-4,3e3)
    plt.legend()


# savedirbello='F:/Balto/'
# for j in range (0,100):
#     spettro=genspettroIRF(cdf, pdf)
#     psdsynth=PSDToSeries(44100*1.6, (IRF[1][0],IRF[1][-1]), spettro)
#     #sf.write(savedirbello+str(j)+'.wav', psdsynth, int(44100/2))
#     nsamples=len(psdsynth)*2
#     sf.write(savedirbello+str(j)+'.wav', sps.resample(psdsynth, nsamples), int(44100))

# savedirbrutto='F:/Brent/'
# for j in range (0,100):
#     spettro=genspettroIRF(cdfcrack, pdfcrack)
#     psdsynthcrack=PSDToSeries(44100*1.6, (IRF[1][0],IRF[1][-1]), spettro)
#     nsamples=len(psdsynthcrack)*2
#     sf.write(savedirbrutto+str(j)+'.wav', sps.resample(psdsynthcrack, nsamples), int(44100))

# for j in range (0,1000):
#     plt.loglog(IRF[1], genspettroIRF(cdf,pdf))
#     plt.ylim(1e-6, 2e4)
# plt.show()

# for j in range (0,100):
#     plt.loglog(IRF[1], genspettroIRF(cdf, pdf))
#     plt.ylim(1e-6, 2e4)
# plt.show()

# for j in range (0,100):
#     plt.loglog(IRF[1], psdarray[j])
#     plt.ylim(1e-6, 2e4)
# plt.show()

# for j in range (0, 100):
#     rand=np.random.random()*33
#     psdbello=plottapsdbucata(pezzobello, (rand, rand+0.2))
# plt.show()