# Experiment about expander
# Chapter 4 of textbook(4.6)

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt


## Generate the sequence of x[n]

# Independent parameters
N = 37
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
plt.axis([0, 150, -0.1, 0.35])
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
# When L = 2
L = 2
NL = N * L - (L - 1)
xe = np.zeros((NL, 1))
for i in range(0, N):
    xe[i * L, 0] = x[i, 0]

# Plot xd[n]
plt.subplot(2, 4, 2)
nn = np.arange(0, NL); nn=nn[: ,np.newaxis]
plt.stem(nn, xe)
plt.axis([0, 150, -0.1, 0.35])
plt.xlabel('n')
plt.ylabel('xe[n]')
plt.title('L = 2')

# Plot amplitude of Fourier transform of xd[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(xe, 1, rr))
plt.subplot(2, 4, 6)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of xe[n]')


##---------------------------##
# When L = 3
L = 3
NL = N * L - (L - 1)
xe = np.zeros((NL, 1))
for i in range(0, N):
    xe[i * L, 0] = x[i, 0]

# Plot xe[n]
plt.subplot(2, 4, 3)
nn = np.arange(0, NL); nn=nn[: ,np.newaxis]
plt.stem(nn, xe)
plt.axis([0, 150, -0.1, 0.35])
plt.xlabel('n')
plt.ylabel('xe[n]')
plt.title('L = 3')

# Plot amplitude of Fourier transform of xe[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(xe, 1, rr))
plt.subplot(2, 4, 7)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of xe[n]')


##---------------------------##
# When L = 4
L = 4
NL = N * L - (L - 1)
xe = np.zeros((NL, 1))
for i in range(0, N):
    xe[i * L, 0] = x[i, 0]

# Plot xe[n]
plt.subplot(2, 4, 4)
nn = np.arange(0, NL); nn=nn[: ,np.newaxis]
plt.stem(nn, xe)
plt.axis([0, 150, -0.1, 0.35])
plt.xlabel('n')
plt.ylabel('xe[n]')
plt.title('L = 4')

# Plot amplitude of Fourier transform of xe[n]
rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
AR = np.absolute(sig.freqz(xe, 1, rr))
plt.subplot(2, 4, 8)
plt.plot(rr / math.pi, AR[1])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude of Fourier transform of xe[n]')
 




plt.show()









