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

a2 = np.reshape(a, (Mc, NH + 1)); a2=np.transpose(a2)
b2 = np.reshape(a, (Ms, NH)); b2=np.transpose(b2)







# ##---------------##
# # Plot group delay
# GD = sig.group_delay((np.flipud(A[:, 0]), A[:, 0]))
# rr = np.linspace(0, math.pi, num=512); rr=rr[:, np.newaxis]
# plt.subplot(1, 2, 1)
# plt.plot(rr / math.pi, GD[1])
# plt.axis([0, 1, 60, 90])
# plt.xlabel('Normalized frequency')
# plt.ylabel('Group delay')
# plt.title('IIR allpass filter')


# # Plot port-zeros
# plt.subplot(1, 2, 2)
# tfx = control.tf(np.flipud(A[:, 0]), A[: , 0])
# control.pzmap(tfx)
# plt.axis([-2, 2, -2, 2])
# plt.grid()
# plt.title('Port-Zero diagram')
# plt.show()