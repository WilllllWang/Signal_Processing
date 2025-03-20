# Design of a case 2 linear-phase FIR bandpass filter

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt



def main():
    N = 40
    Ws1 = 0.3 * math.pi 
    Wp1 = 0.4 * math.pi
    Wp2 = 0.65 * math.pi
    Ws2 = 0.75 * math.pi
    Ns = 200

    NH = N // 2
    deltaW = math.pi / Ns
    Ns_p = round((Wp2 - Wp1) / deltaW)
    Ns_s1 = round(Ws1 / deltaW)
    Ns_s2 = round((math.pi - Ws2) / deltaW) 
    NV = np.arange(1, NH + 1); NV=NV[: , np.newaxis]

    P = np.zeros((NH, 1))
    Qp = np.zeros((NH, NH)) 
    Qs1 = np.zeros((NH, NH))
    Qs2 = np.zeros((NH, NH)) 

    # Passband
    for iw in range(0, Ns_p + 1):
        w = Wp1 + iw * deltaW
        P -= 2 * np.cos((NV - 0.5) * w)
        Qp += np.cos((NV - 0.5) * w) @ np.transpose(np.cos((NV - 0.5) * w))
    P *= (Wp2 - Wp1) / (Ns_p + 1)
    Qp *= (Wp2 - Wp1) / (Ns_p + 1)

    # Stopband 1
    for iw in range(0, Ns_s1 + 1):
        w = iw * deltaW
        Qs1 += np.cos((NV - 0.5) * w) @ np.transpose(np.cos((NV - 0.5) * w))
    Qs1 *= Ws1 / (1 + Ns_s1)

    # Stopband 2
    for iw in range(0, Ns_s2 + 1):
        w = Ws2 + iw * deltaW
        Qs2 += np.cos((NV - 0.5) * w) @ np.transpose(np.cos((NV - 0.5) * w))
    Qs2 *= (math.pi - Ws2) / (1 + Ns_s2)

    Q = Qp + Qs1 + Qs2
    A = -0.5 * np.linalg.inv(Q) @ P


    ##---------------------------##
    # Convert A to h
    h = np.zeros((N, 1))
    h[0: NH, 0] = 0.5 * np.flipud(A[:, 0])
    h[NH: N, 0] = 0.5 * A[:, 0]

    # Plot input response
    plt.subplot(2, 2, 1)
    nn = np.arange(0, N); nn=nn[: ,np.newaxis]
    plt.stem(nn, h)
    plt.axis([0, N, -0.5, 0.5])
    plt.xlabel('n')
    plt.ylabel('Input response')
    plt.title('Case 2 bandpass filter')
    
    # Plot amplitude response
    rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
    AR = np.absolute(sig.freqz(h, 1, rr))
    plt.subplot(2, 2, 2)
    plt.plot(rr / math.pi, AR[1])
    plt.axis([0, 1, 0, 1.1])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('Case 2 bandpass filter')

    # Signal simulation
    nnn = np.arange(0, 201); nnn=nnn[:, np.newaxis]
    w1 = 0.15 * math.pi
    w2 = 0.5 * math.pi
    w3 = 0.85 * math.pi
    x = np.cos(w1 * nnn) + np.sin(w2 * nnn) + np.sin(w3 * nnn)
    plt.subplot(2, 2, 3)
    plt.plot(nnn, x)
    plt.axis([0, 200, -3, 3])
    plt.xlabel('n')
    plt.ylabel('Input x[n]')
    
    y = sig.lfilter(h[:, 0], 1, x[:, 0])
    plt.subplot(2, 2, 4)
    plt.plot(nnn, y)
    plt.axis([0, 200, -3, 3])
    plt.xlabel('n')
    plt.ylabel('Output y[n]')

    
    plt.show()





if __name__ == "__main__":
    main()



















