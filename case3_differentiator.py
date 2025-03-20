# Design of a case 3 differentiator

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt



def main():
    N = 41
    Wp = 0.9 * math.pi
    Ns = 200

    NH = (N - 1) // 2
    deltaW = math.pi / Ns
    Ns_p = round(Wp / deltaW)
    NV = np.arange(1, NH + 1); NV=NV[: , np.newaxis]

    P = np.zeros((NH, 1))
    Qp = np.zeros((NH, NH)) # Passband

    # Passband
    for iw in range(0, Ns_p + 1):
        w = iw * deltaW
        P -= 2 * w * np.sin(NV * w)
        Qp += np.sin(NV * w) @ np.transpose(np.sin(NV * w))
    P *= Wp / (Ns_p + 1)
    Qp *= Wp / (Ns_p + 1)

    Q = Qp 
    A = -0.5 * np.linalg.inv(Q) @ P


    ##---------------------------##
    # Convert A to h
    h = np.zeros((N, 1))
    h[0: NH, 0] = 0.5 * np.flipud(A[0: NH, 0])
    h[NH + 1: N, 0] = -0.5 * A[0: NH, 0]

    # Plot input response
    plt.subplot(1, 2, 1)
    nn = np.arange(0, N); nn=nn[: ,np.newaxis]
    plt.stem(h)
    plt.axis([0, N, -1.1, 1.1])
    plt.xlabel('n')
    plt.ylabel('Input response')
    plt.title('Case 3 differentiator')
    
    # Plot amplitude response
    rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
    AR = np.absolute(sig.freqz(h, 1, rr))
    plt.subplot(1, 2, 2)
    plt.plot(rr / math.pi, AR[1])
    plt.axis([0, 1, 0, 3.5])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('Case 3 differentiator')

    
    plt.show()











if __name__ == "__main__":
    main()