# Experiment about compressor
# Chapter 4 of textbook(4.6)

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt


## Generate the sequence of x[n]

# Independent parameters
N = 151
Wp = 0.25 * math.pi 
Ws = 0.3 * math.pi
Ns = 2000

# Dependent parameters
NH = (N - 1) // 2
deltaW = math.pi / Ns
Ns_p = round(Wp / deltaW)
Ns_s = round((math.pi - Ws) / deltaW) 
NV = np.arange(0, NH + 1); NV=NV[: , np.newaxis]


##---------------------------##
P = np.zeros((NH + 1, 1))
Qp = np.zeros((NH + 1, NH + 1)) 
Qs = np.zeros((NH + 1, NH + 1)) 

# Passband
for iw in range(0, Ns_p + 1):
    w = iw * deltaW
    P -= 2 * np.cos(NV * w)
    Qp += np.cos(NV * w) @ np.transpose(np.cos(NV * w))
P *= Wp / (Ns_p + 1)
Qp *= Wp / (Ns_p + 1)

# Stopband
for iw in range(0, Ns_s + 1):
    w = Ws + iw * deltaW
    Qs += np.cos(NV * w) @ np.transpose(np.cos(NV * w))
Qs *= (math.pi - Ws) / (1 + Ns_s)

Q = Qp + Qs
A = -0.5 * np.linalg.inv(Q) @ P


##---------------------------##
# Convert A to h
h = np.zeros((N, 1))
h[NH, 0] = A[0, 0]
h[0: NH, 0] = 0.5 * np.flipud(A[1: NH + 1, 0])
h[NH + 1: N, 0] = 0.5 * A[1: NH + 1, 0]

# Get x
x = h.copy()


# Plot x[n]
plt.subplot(2, 4, 1)
nn = np.arange(0, N); nn=nn[: ,np.newaxis]
plt.stem(nn, x)
plt.axis([0, 150, -0.1, 0.3])
plt.xlabel('n')
plt.ylabel('x[n]')

# Plot amplitude of Fourier transform of x[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(x, 1, rr))
plt.subplot(2, 4, 5)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of x[n]')


##---------------------------##
# When M = 2
M = 2
NM = math.ceil(N / M)
xd = np.zeros((NM, 1))
for i in range(0, NM):
    xd[i, 0] = x[i * M, 0]

# Plot xd[n]
plt.subplot(2, 4, 2)
nn = np.arange(0, NM); nn=nn[: ,np.newaxis]
plt.stem(nn, xd)
plt.axis([0, 150, -0.1, 0.3])
plt.xlabel('n')
plt.ylabel('xd[n]')
plt.title('M = 2')

# Plot amplitude of Fourier transform of xd[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(xd, 1, rr))
plt.subplot(2, 4, 6)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of xd[n]')


##---------------------------##
# When M = 3
M = 3
NM = math.ceil(N / M)
xd = np.zeros((NM, 1))
for i in range(0, NM):
    xd[i, 0] = x[i * M, 0]

# Plot xd[n]
plt.subplot(2, 4, 3)
nn = np.arange(0, NM); nn=nn[: ,np.newaxis]
plt.stem(nn, xd)
plt.axis([0, 150, -0.1, 0.3])
plt.xlabel('n')
plt.ylabel('xd[n]')
plt.title('M = 3')

# Plot amplitude of Fourier transform of xd[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(xd, 1, rr))
plt.subplot(2, 4, 7)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of xd[n]')


##---------------------------##
# When M = 4, the signal is different at high frequency
# So don't compress too much
M = 4
NM = math.ceil(N / M)
xd = np.zeros((NM, 1))
for i in range(0, NM):
    xd[i, 0] = x[i * M, 0]


# Plot xd[n]
plt.subplot(2, 4, 4)
nn = np.arange(0, NM); nn=nn[: ,np.newaxis]
plt.stem(nn, xd)
plt.axis([0, 150, -0.1, 0.3])
plt.xlabel('n')
plt.ylabel('xd[n]')
plt.title('M = 4')

# Plot amplitude of Fourier transform of xd[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(xd, 1, rr))
plt.subplot(2, 4, 8)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of xd[n]')







plt.show()









