# Design of Variable Fractional Delay digital filter

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt

##---------------##
# Independent parameters
N = 50
M = 7
Wp = 0.9 * math.pi
Nw = 200
Np = 60

# Dependent parameters
NH = N // 2
deltaW = Wp / Nw
deltaP = 1 / Np
Mc = M // 2
Ms = math.ceil(M / 2)
nma = Mc * (NH + 1)
nmb = Ms * NH
NV = np.arange(0, NH + 1); NV = NV[:, np.newaxis]
NV1 = np.arange(1, NH + 1); NV1 = NV1[:, np.newaxis]
Nwp = (Nw + 1) * (Np + 1)

##---------------##
# Calculate a
ra = np.zeros((nma, 1))
Qa = np.zeros((nma, nma))
for iw in range(0, Nw + 1):
    w = iw * deltaW
    for ip in range(0, Np + 1):
        p = -0.5 + ip * deltaP
        cwp = np.zeros((nma, 1))
        for im in range(1, Mc + 1):
            cwp[(im - 1) * (NH + 1): im * (NH + 1), 0] = p**(2 * im) * np.cos(NV[:, 0] * w)
        ra = ra - 2 * (np.cos(p * w) - 1) * cwp
        Qa = Qa + cwp * np.transpose(cwp)

ra = (ra * Wp) / Nwp 
Qa = (Qa * Wp) / Nwp
a = (-0.5 * np.linalg.inv(Qa)) @ ra

# Calculate b
rb = np.zeros((nmb, 1))
Qb = np.zeros((nmb, nmb))
for iw in range(0, Nw + 1):
    w = iw * deltaW
    for ip in range(0, Np + 1):
        p = -0.5 + ip * deltaP
        swp = np.zeros((nmb, 1))
        for im in range(1, Ms + 1):
            swp[(im - 1) * NH: im * NH, 0] = p**(2 * im - 1) * np.sin(NV1[:, 0] * w)
        rb = rb + 2 * np.sin(p * w) * swp
        Qb = Qb + swp * np.transpose(swp)

rb = (rb * Wp) / Nwp 
Qb = (Qb * Wp) / Nwp
b = (-0.5 * np.linalg.inv(Qb)) @ rb

a2 = np.reshape(a, (Mc, NH + 1)); a2 = np.transpose(a2)
b2 = np.reshape(b, (Ms, NH)); b2 = np.transpose(b2)
h2 = np.zeros((N + 1, M + 1))

# when m=0
h2[NH, 0] = 1

# when m:even
for im in range (1, Mc + 1):
    h2[NH, 2 * im] = a2[0, im - 1]
    h2[0: NH, 2 * im] = 0.5 * np.flipud(a2[1: NH + 1, im - 1])
    h2[NH + 1: , 2 * im] = 0.5 * a2[1: , im - 1]

# m:odd
for im in range (1, Ms + 1):
    h2[0: NH, 2*im - 1]= 0.5 * np.flipud(b2[:, im - 1])
    h2[NH + 1:, 2*im - 1] = -0.5 * b2[:, im - 1]

##---------------##
# Plot Amplitude response and group delay
MR = np.zeros((Nw + 1, Np + 1, 1))
GD = np.zeros((512, Np + 1))
h = np.zeros((N + 1, 1))
for ip in range (0, Np + 1):
    p = -0.5 + ip * deltaP
    h = h2[:, 0]
    for im in range (1, M + 1): 
        h = h + (p**im) * h2[:, im]
    rr = np.linspace(0, Wp, num=Nw + 1); rr = rr[:, np.newaxis]
    AR = np.absolute(sig.freqz(h, 1, rr))
    MR[:, ip] = AR[1]

    GDD = sig.group_delay((h, 1))
    GD[:, ip] = GDD[1]

# Amplitude response
plt.subplot(1, 2, 1)
for ip in range (0, Np + 1): 
    plt.plot(rr / math.pi, MR[:, ip])
plt.axis([0, Wp / math.pi, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude response')
plt.title('VFD')

# Group delay
plt.subplot(1, 2, 2)
rr = np.linspace(0, math.pi, num=512); rr = rr[:, np.newaxis]
for ip in range (0, Np + 1): 
    plt.plot(rr / math.pi, GD[:, ip])
plt.axis([0, Wp / math.pi, NH - 1, NH + 1])
plt.xlabel('Normalized frequency')
plt.ylabel('Group delay')
plt.title('VFD')  

plt.show()