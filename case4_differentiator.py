# Design of a case 4 differentiator

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt



def main():
    N = 30
    Ns = 200

    NH = N // 2
    deltaW = math.pi / Ns
    Ns = round(math.pi / deltaW)
    NV = np.arange(1, NH + 1); NV=NV[: , np.newaxis]

    P = np.zeros((NH, 1))
    Q = np.zeros((NH, NH)) 

    # Passband
    for iw in range(0, Ns + 1):
        w = iw * deltaW
        P -= 2 * w * np.sin((NV - 0.5) * w)
        Q += np.sin((NV - 0.5) * w) @ np.transpose(np.sin((NV - 0.5) * w))
    P *= math.pi / (Ns + 1)
    Q *= math.pi / (Ns + 1)


    A = -0.5 * np.linalg.inv(Q) @ P


    ##---------------------------##
    # Convert A to h
    h = np.zeros((N, 1))
    h[0: NH, 0] = 0.5 * np.flipud(A[:, 0])
    h[NH: N, 0] = -0.5 * A[:, 0]

    # Plot Impulse response
    plt.subplot(1, 2, 1)
    nn = np.arange(0, N); nn=nn[: ,np.newaxis]
    plt.stem(nn, h)
    plt.axis([0, N, -1.5, 1.5])
    plt.xlabel('n')
    plt.ylabel('Impulse response')
    plt.title('Case 4 differentiator')
    
    # Plot amplitude response
    rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
    AR = np.absolute(sig.freqz(h, 1, rr))
    plt.subplot(1, 2, 2)
    plt.plot(rr / math.pi, AR[1])
    plt.axis([0, 1, 0, 3.5])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('Case 4 differentiator')
    
    
    plt.show()





if __name__ == "__main__":
    main()



















