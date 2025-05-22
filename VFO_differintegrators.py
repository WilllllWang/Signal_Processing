# Design of Variable Fractional Order (VFO) differintegrators

import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt


def main():
    # Case 1:
    VFO(40, 5, 0.05 * math.pi, 0.95 * math.pi, -0.5, 0.5)

    # Case 2:
    VFO(30, 6, 0, 0.9 * math.pi, 1, 2)

    # Case 3:
    VFO(60, 6, 0.05 * math.pi, 0.9 * math.pi, -1.5, -0.5)





def VFO(N, M, w1, w2, p1, p2):
    # Independent parameters
    Nw = 200
    Np = 60
    
    # Dependent parameters
    NH = N // 2
    nma = (NH + 1) * (M + 1)
    nmb = NH * (M + 1)
    deltaW = (w2 - w1) / Nw
    deltaP = (p2 - p1) / Np
    Nwp = (Nw + 1) * (Np + 1)
    NVa = np.arange(0, NH + 1); NVa = NVa[:, np.newaxis]
    NVb = np.arange(1, NH + 1); NVb = NVb[:, np.newaxis]


    ## -------- Calculate a --------
    ra = np.zeros((nma, 1))
    Qa = np.zeros((nma, nma))
    for ip in range(0, Np + 1):
        p = p1 + ip * deltaP
        for iw in range(0, Nw + 1):
            w = w1 + iw * deltaW
            cwp = np.zeros((nma, 1))
            for im in range(0, M + 1):
                cwp[im * (NH + 1): (im + 1) * (NH + 1), 0] = (p**im) * np.cos(w * NVa[:, 0])
            ra = ra - 2 * (w**p) * np.cos(p * math.pi / 2) * cwp
            Qa = Qa + cwp @ np.transpose(cwp)

    ra = ra * (w2 - w1) * (p2 - p1) / Nwp
    Qa = Qa * (w2 - w1) * (p2 - p1) / Nwp 
    a = -0.5 * np.linalg.inv(Qa) @ ra  


    ## -------- Calculate b --------
    rb = np.zeros((nmb, 1))
    Qb = np.zeros((nmb, nmb))
    for ip in range(0, Np + 1):
        p = p1 + ip * deltaP
        for iw in range(0, Nw + 1):
            w = w1 + iw * deltaW
            swp = np.zeros((nmb, 1))
            for im in range(0, M + 1):
                swp[im * NH: (im + 1) * NH, 0] = (p**im) * np.sin(w * NVb[:, 0])
            rb = rb - 2 * (w**p) * np.sin(p * math.pi / 2) * swp
            Qb = Qb + swp @ np.transpose(swp)

    rb = rb * (w2 - w1) * (p2 - p1) / Nwp 
    Qb = Qb * (w2 - w1) * (p2 - p1) / Nwp 
    b = -0.5 * np.linalg.inv(Qb) @ rb  


    # Reshape and get h2
    a2 = np.reshape(a, (M + 1, NH + 1)); a2 = np.transpose(a2)
    he = np.zeros((N + 1, M + 1))
    he[NH, :] = a2[0, :]
    he[0: NH, :] = 0.5 * np.flipud(a2[1: NH + 1, :])
    he[NH + 1: N + 1, :] = 0.5 * a2[1: NH + 1, :]

    b2 = np.reshape(b, (M + 1, NH)); b2 = np.transpose(b2)
    ho = np.zeros((N + 1, M + 1))
    ho[0: NH, :] = 0.5 * np.flipud(b2)
    ho[NH + 1: N + 1, :] = -0.5 * b2

    h2 = he + ho


    ##---------------##
    # Plot Amplitude response 
    MR = np.zeros((Nw + 1, Np + 1, 1))
    for ip in range (0, Np + 1):
        p = p1 + ip * deltaP
        h = h2[:, 0]
        for im in range (1, M + 1): 
            h = h + h2[:, im] * (p**im)
        rr = np.linspace(w1, w2, num=Nw+1); rr = rr[:, np.newaxis]
        MRR = np.absolute(sig.freqz(h, 1, rr))
        MR[:, ip] = MRR[1]

    # Amplitude response
    for ip in range (0, Np + 1): 
        plt.plot(rr / math.pi, MR[:, ip])
    plt.axis([w1 / math.pi, w2 / math.pi, 0, 3])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('VFO')

    plt.show()







if __name__ == "__main__":
    main()