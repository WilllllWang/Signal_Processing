# Design of a case 1 linear-phase FIR lowpass/highpass filter

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt



def main():
    N = 31
    Wp = 0.4 * math.pi 
    Ws = 0.5 * math.pi
    Ns = 200

    # To calculate integrals, sample data points then get the average and multiply by the range 
    NH = (N - 1) // 2
    deltaW = math.pi / Ns
    Ns_p = round(Wp / deltaW)
    Ns_s = round((math.pi - Ws) / deltaW) 
    NV = np.arange(0, NH + 1); NV=NV[: , np.newaxis]

    P = np.zeros((NH + 1, 1))
    Qp = np.zeros((NH + 1, NH + 1)) # Passband
    Qs = np.zeros((NH + 1, NH + 1)) # Stopband

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

    # Plot input response
    plt.subplot(2, 2, 1)
    nn = np.arange(0, N); nn=nn[: ,np.newaxis]
    plt.stem(h)
    plt.axis([0, N, -0.1, 0.5])
    plt.xlabel('n')
    plt.ylabel('Input response')
    plt.title('Case 1 lowpass filter')
    
    # Plot amplitude response
    rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
    AR = np.absolute(sig.freqz(h, 1, rr))
    plt.subplot(2, 2, 2)
    plt.plot(rr / math.pi, AR[1])
    plt.axis([0, 1, 0, 1.1])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('Case 1 lowpass filter')

    # Signal simulation
    nnn = np.arange(0, 201); nnn=nnn[:, np.newaxis]
    w1 = 0.3 * math.pi
    w2 = 0.7 * math.pi
    x = np.cos(w1 * nnn) + np.sin(w2 * nnn)
    plt.subplot(2, 2, 3)
    plt.plot(nnn, x)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('Input x[n]')
    
    # Lowpass filter
    y = sig.lfilter(h[:, 0], 1, x[:, 0])
    plt.subplot(2, 2, 4)
    plt.plot(nnn, y)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('Output y[n]')

    
    plt.show()


    ##---------------------------------------##
    # Convert lowpass filter to highpass filter
    for i in range(0, N):
        h[i, 0] = ((-1) ** i) * h[i, 0]

    # Plot input response
    plt.subplot(2, 2, 1)
    plt.stem(nn, h)
    plt.axis([0, N, -0.5, 0.5])
    plt.xlabel('n')
    plt.ylabel('Input response')
    plt.title('Case 1 highpass filter')

    # Plot amplitude response
    AR = np.absolute(sig.freqz(h, 1, rr))
    plt.subplot(2, 2, 2)
    plt.plot(rr / math.pi, AR[1])
    plt.axis([0, 1, 0, 1.1])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('Case 1 highpass filter')

    # Signal simulation
    plt.subplot(2, 2, 3)
    plt.plot(nnn, x)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('Input x[n]')
    
    # Lowpass filter
    y = sig.lfilter(h[:, 0], 1, x[:, 0])
    plt.subplot(2, 2, 4)
    plt.plot(nnn, y)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('Output y[n]')

    
    plt.show()











if __name__ == "__main__":
    main()