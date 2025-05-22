# Design of Variable lowpass FIR filter

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt


def main():
    VLP(50, 7, 0.2 * math.pi, 0.6 * math.pi, 0.2 * math.pi)


def VLP(N, M, wp1, wp2, wt):
    # Independent parameters
    Nw = 200
    Np = 60
    
    # Dependent parameters
    NH = N // 2
    nm = (NH + 1) * (M + 1)
    deltaW = math.pi / Nw
    deltaP = 1 / Np
    NV = np.arange(0, NH + 1); NV = NV[:, np.newaxis]


    ##--------------------##
    # Passband and Stopband
    r = np.zeros((nm, 1))
    Qp = np.zeros((nm, nm))
    Qs = np.zeros((nm, nm))
    Nsp = 0
    Nss = 0
    for ip in range(0, Np + 1):
        p = -0.5 + ip * deltaP
        for iw in range(0, Nw + 1):
            w = iw * deltaW
            wp = (p + 0.5) * (wp2 - wp1) + wp1
            cwp = np.zeros((nm, 1))
            for im in range(0, M + 1):
                cwp[im * (NH + 1): (im + 1) * (NH + 1), 0] = (p**im) * np.cos(w * NV[:, 0])

            if w <= wp:
                Nsp += 1 
                r = r - 2 * cwp
                Qp = Qp + cwp @ np.transpose(cwp)
            elif w >= (wp + wt):
                Nss += 1
                Qs = Qs + cwp @ np.transpose(cwp)            

    r = (0.5 * (wp1 + wp2) * r) / Nsp 
    Qp = (0.5 * (wp1 + wp2) * Qp) / Nsp
    Qs = (0.5 * ((math.pi - wp1 - wt) + (math.pi - wp2 - wt)) * Qs) / Nss 
    a = -0.5 * np.linalg.inv(Qp + Qs) @ r

    # Reshape and get a2 then h2
    a2 = np.reshape(a, (M + 1, NH + 1)); a2 = np.transpose(a2)
    h2 = np.zeros((N + 1, M + 1))
    h2[NH, :] = a2[0, :]
    h2[0: NH, :] = 0.5 * np.flipud(a2[1: NH + 1, :])
    h2[NH + 1: N + 1, :] = 0.5 * a2[1: NH + 1, :]


    ##--------------------##
    # Plot Amplitude response 
    MR = np.zeros((Nw + 1, Np + 1, 1))
    for ip in range (0, Np + 1):
        p = -0.5 + ip * deltaP
        h = h2[:, 0]
        for im in range (1, M + 1): 
            h = h + h2[:, im] * (p**im)
        rr = np.linspace(0, math.pi, num=Nw+1); rr = rr[:, np.newaxis]
        MRR = np.absolute(sig.freqz(h, 1, rr))
        MR[:, ip] = MRR[1]

    # Amplitude response
    for i in range (0, Np + 1): 
        plt.plot(rr / math.pi, MR[:, i])
    plt.axis([0, 1, 0, 1.1])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('VLP')

    plt.show()







if __name__ == "__main__":
    main()