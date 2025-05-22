# Design of IIR allpass filters

import math
import control
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt

##---------------##
# Independent parameters
N = 70
Wp1 = 0.08 * math.pi
Wp2 = 0.12 * math.pi
Wp3 = 0.685 * math.pi
Wp4 = 0.715 * math.pi
Ns = 10000

# Dependent parameters
deltaW = math.pi / Ns
NV = np.arange(0, N + 1); NV=NV[:, np.newaxis]


##---------------##
# Main
PHd = np.zeros((Ns + 1, 1))
PRd = np.zeros((Ns + 1, 1))
W = np.zeros((Ns + 1, 1))


for iw in range(0, Ns + 1):
    w = iw * deltaW
    if w <= Wp1:
        PHd[iw, 0] = -85 * w
        W[iw, 0] = 90
    elif Wp2 <= w <= Wp3:
        PHd[iw, 0] = (-65 * w) - (2 * math.pi)
        W[iw, 0] = 1
    elif Wp4 <= w:
        PHd[iw, 0] = (-75 * w) + (5 * math.pi)
        W[iw, 0] = 30

    PRd[iw, 0] = -0.5 * (N * w + PHd[iw, 0])


##---------------##
# Calculate Error function = transpose(A) * Q * A
Q1 = np.zeros((N + 1, N + 1))
Q2 = np.zeros((N + 1, N + 1))
Q3 = np.zeros((N + 1, N + 1))

s1 = 0
s2 = 0
s3 = 0

for iw in range(0, Ns + 1):
    w = iw * deltaW
    C = np.sin(PRd[iw, 0]) * np.cos(NV[:, 0] * w) + np.cos(PRd[iw, 0]) * np.sin(NV[:, 0] * w)
    C = C[:, np.newaxis]
    if w <= Wp1:
        s1 += 1
        Q1 = Q1 + W[iw, 0] * C @ np.transpose(C)
    elif Wp2 <= w <= Wp3:
        s2 += 1
        Q2 = Q2 + W[iw, 0] * C @ np.transpose(C)
    elif Wp4 <= w:
        s3 += 1
        Q3 = Q3 + W[iw, 0] * C @ np.transpose(C)

Q1 *= Wp1 / s1
Q2 *= (Wp3 - Wp2) / s2
Q3 *= (math.pi - Wp4) / s3
Q = Q1 + Q2 + Q3

[D, X] = np.linalg.eig(Q)
D = np.real(D)
idx = np.argmin(D)
A = np.real(X[:, idx]); A=A[:, np.newaxis]


##---------------##
# Plot group delay
GD = sig.group_delay((np.flipud(A[:, 0]), A[:, 0]))
rr = np.linspace(0, math.pi, num=512); rr=rr[:, np.newaxis]
plt.subplot(1, 2, 1)
plt.plot(rr / math.pi, GD[1])
plt.axis([0, 1, 60, 90])
plt.xlabel('Normalized frequency')
plt.ylabel('Group delay')
plt.title('IIR allpass filter')


# Plot port-zeros
plt.subplot(1, 2, 2)
tfx = control.tf(np.flipud(A[:, 0]), A[: , 0])
control.pzmap(tfx)
plt.axis([-2, 2, -2, 2])
plt.grid()
plt.title('Port-Zero diagram')
plt.show()