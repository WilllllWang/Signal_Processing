# Design of a two channel filter bank
# Textbook 4.7.6
import math
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt




def two_channel_filter_bank():
    # Independent parameters
    N = 50
    Ws = 0.55 * math.pi
    Ns = 400

    # Design prototype of type 2 lowpass filter
    # Dependent parameters
    Wp = math.pi - Ws
    NH = N // 2
    deltaW = math.pi / Ns
    Ns_p = round(Wp / deltaW)
    Ns_s = round((math.pi - Ws) / deltaW)
    NV = np.arange(1, NH + 1); NV=NV[: , np.newaxis]

    P = np.zeros((NH, 1))
    Qp = np.zeros((NH, NH)) 
    Qs = np.zeros((NH, NH))

    # Passband
    for iw in range(0, Ns_p + 1):
        w = iw * deltaW
        P -= 2 * np.cos((NV - 0.5) * w)
        Qp += np.cos((NV - 0.5) * w) @ np.transpose(np.cos((NV - 0.5) * w))
    P = Wp * P / (Ns_p + 1)
    Qp = Wp * Qp / (Ns_p + 1)

    # Stopband 
    for iw in range(0, Ns_s + 1):
        w = Ws + iw * deltaW
        Qs += np.cos((NV - 0.5) * w) @ np.transpose(np.cos((NV - 0.5) * w))
    Qs = (math.pi - Ws) * Qs / (1 + Ns_s)


    Q = Qp + Qs
    A = -0.5 * np.linalg.inv(Q) @ P


    ##---------------------------##
    # Convert A to h
    h = np.zeros((N, 1))
    h[0: NH, 0] = 0.5 * np.flipud(A[:, 0])
    h[NH: N, 0] = 0.5 * A[:, 0]

    # Plot amplitude response
    # rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
    # AR0 = np.absolute(sig.freqz(h, 1, rr))
    # plt.plot(rr / math.pi, AR0[1])
    # plt.axis([0, 1, 0, 1.1])
    # plt.xlabel('Normalized frequency')
    # plt.ylabel('Amplitude response')
    # plt.title('Prototype lowpass filter')


    ##---------------------------##
    # Iterative design
    # Independent parameters
    alpha = 0.1
    beta = 0.5
    deltaK = 1000 
    epsilon = 0.00001
    iter = 0
    while (deltaK > epsilon):
        iter += 1
        print(f"Iteration {iter}")

        # Plot amplitude response
        rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
        AR0 = np.absolute(sig.freqz(h, 1, rr))
        plt.subplot(2, 2, 1)
        plt.plot(rr / math.pi, AR0[1])
        plt.axis([0, 1, 0, 1.1])
        plt.xlabel('Normalized frequency')
        plt.ylabel('Amplitude response')
        plt.title('Prototype lowpass filter')

        Ak1 = A
        P = np.zeros((NH, 1))
        Q1 = np.zeros((NH, NH))
        for iw in range(0, Ns + 1):
            w = iw * deltaW
            C = np.cos((NV - 0.5) * w)
            C_pi = np.cos((NV - 0.5) * (w - math.pi))
            CC = (np.transpose(Ak1) @ C) * C + (np.transpose(Ak1) @ C_pi) * C_pi
            P = P - 2 * CC
            Q1 = Q1 + CC @ np.transpose(CC)
        P = math.pi * P / (Ns + 1)
        Q1 = math.pi * Q1 / (Ns + 1)

        Q2 = np.zeros((NH, NH))
        for iw in range(0, Ns_s + 1):
            w = Ws + iw * deltaW
            C = np.cos((NV - 0.5) * w)
            Q2 = Q2 + C @ np.transpose(C)
        Q2 = alpha * (math.pi - Ws) * Q2 / (Ns_s + 1)

        A = -0.5 * np.linalg.inv(Q1 + Q2) @ P
        A = beta * A + (1 - beta) * Ak1 # Slight fixes
        # Convert A to h
        h0 = np.zeros((N, 1))
        h0[0: NH, 0] = 0.5 * np.flipud(A[:, 0])
        h0[NH: N, 0] = 0.5 * A[:, 0]

        # Plot amplitude response
        rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
        AR = np.absolute(sig.freqz(h0, 1, rr))
        plt.subplot(2, 2, 2)
        plt.plot(rr / math.pi, AR[1])
        plt.axis([0, 1, 0, 1.1])
        plt.xlabel('Normalized frequency')
        plt.ylabel('Amplitude response')
        plt.title('Designed lowpass filter')
        
        h1 = np.zeros((N, 1))
        for i in range(0, N):
            h1[i, 0] = (-1)**i * h0[i, 0]


        # Plot Transmission t / Overall impulse response
        t = sig.convolve(h0, h0) - sig.convolve(h1, h1) 
        nn = np.arange(0, 2*N - 1); nn=nn[: ,np.newaxis]
        plt.subplot(2, 2, 3)
        plt.stem(nn, t)
        plt.xlabel('n')
        plt.ylabel('Overall impulse response')

        # Plot Overall amplitude response
        rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
        ARt = np.absolute(sig.freqz(t, 1, rr))
        plt.subplot(2, 2, 4)
        plt.plot(rr / math.pi, ARt[1])
        plt.axis([0, 1, 0, 2])
        plt.xlabel('Normalized frequency')
        plt.ylabel('Overall amplitude response')
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        deltaK = np.linalg.norm(A - Ak1) / np.linalg.norm(A)


    ##---------------------##
    # Signal simulation
    g0 = 2 * h0.copy()
    g1 = -2 * h1.copy()

    nnn = np.arange(0, 201); nnn=nnn[:, np.newaxis]
    w1 = 0.3 * math.pi
    w2 = 0.7 * math.pi
    x = np.cos(w1 * nnn) + np.sin(w2 * nnn) 
    plt.subplot(3, 3, 1)
    plt.plot(nnn, x)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('x')
    
    xh0 = sig.lfilter(h0[:, 0], 1, x[:, 0]); xh0=xh0[:, np.newaxis]
    plt.subplot(3, 3, 2)
    plt.plot(nnn, xh0)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('xh[0]')

    xh1 = sig.lfilter(h1[:, 0], 1, x[:, 0]); xh1=xh1[:, np.newaxis]
    plt.subplot(3, 3, 3)
    plt.plot(nnn, xh1)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('xh[1]')

    # d2 = downsampling, u2 = upsampling
    xh0d2u2 = xh0.copy()
    for i in range(0, 201):
        if i % 2 == 1:
            xh0d2u2[i, 0] = 0
    
    xh1d2u2 = xh1.copy()
    for i in range(0, 201):
        if i % 2 == 1:
            xh1d2u2[i, 0] = 0

    




    plt.show()
    













if __name__ == "__main__":
    two_channel_filter_bank()



















