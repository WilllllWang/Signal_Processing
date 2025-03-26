import sys
import math
import control
import numpy as np
from scipy import signal as sig
from scipy import fft 
from matplotlib import pyplot as plt
from qpsolvers import solve_qp

def main():
    # First part calculates x
    # Second part plots the responses
    # firstPart()
    secondPart()



##--------------------------------------------##
##-----------------First part-----------------##
def firstPart():
    # Independent parameters
    n = 15
    m = 30
    tau = 20
    Wp = 0.5 * math.pi
    Ws = 0.55 * math.pi
    Ns = 10000
    M = 100
    delta = 0.00001
    epsilon = 0.00001

    # Dependent parameters
    j = complex(0, 1)
    nm = n + m + 1
    deltaW = math.pi / Ns
    Ns_p = round(Wp / deltaW)
    Ns_s = round((math.pi - Ws) / deltaW)
    deltaM = math.pi / M
    NV = np.arange(1, n + 1); NV=NV[:, np.newaxis] 
    MV = np.arange(0, m + 1); MV=MV[:, np.newaxis]

    # Main
    x = np.zeros((nm, 1))
    deltaX = 1000
    iter = 0

    while (deltaX > epsilon):
        iter += 1
        print(f"Iteration {iter}")
        
        xp = x.copy()
        a = x[0: n, 0]; a=a[:, np.newaxis]

        # Get Q
        # Stopband
        Qs = np.zeros((nm, nm)) 
        for iw in range(0, Ns_s + 1):
            w = Ws + iw * deltaW
            eaw = np.exp(-j * NV * w)
            WH = 1 / (abs(1 + np.transpose(a) @ eaw)) ** 2 

            ew = np.vstack((np.zeros((n, 1)), -np.exp(-j * MV * w))) 
            Qs = Qs + WH * (ew @ np.transpose(np.conjugate(ew)))

        Qs *= (math.pi - Ws) / (Ns_s + 1)

        # Passband
        Qp = np.zeros((nm, nm)) 
        r = np.zeros((nm , 1))
        for iw in range(0, Ns_p + 1):
            w = iw * deltaW
            eaw = np.exp(-j * NV * w)
            WH = 1 / (abs(1 + np.transpose(a) @ eaw)) ** 2
            
            ew = np.vstack((np.exp(-j * tau * w) * eaw, -np.exp(-j * MV * w))) 
            r = r + WH * np.real(np.conjugate(np.exp(-j * tau * w)) * ew)
            Qp = Qp + WH * ew @ np.transpose(np.conjugate(ew))

        r *= Wp / (Ns_p + 1) # Get average
        Qp *= Wp / (Ns_p + 1)

        
        Q = np.real(Qp + Qs) 

        B = np.zeros((M + 1, nm))
        for iw in range(0, M + 1):
            w = iw * deltaM
            B[iw, 0: n] = -np.cos(np.transpose(NV) * w) 

        d = (1 - delta) * np.ones((M + 1, 1))

        # Solve
        x = solve_qp(Q, r, B, d, solver = 'quadprog'); x=x[:, np.newaxis]
        a = x[0: n, 0]; a=a[:, np.newaxis]
        b = x[n: nm, 0]; b=b[:, np.newaxis]
        print(f"a = {a}\n\nb = {b}")

        deltaX = np.linalg.norm(x - xp) / np.linalg.norm(x)
        print(f"deltaX = {deltaX}\n") 

    np.save("IIR_lowpass_filter.npy", x)



##--------------------------------------------##
##----------------Second part-----------------##
def secondPart():
    # Independent parameters
    n = 15
    m = 30
    Ns = 10000

    # Dependent parameters
    nm = n + m + 1
    NV = np.arange(1, n + 1); NV=NV[:, np.newaxis] 
    MV = np.arange(0, m + 1); MV=MV[:, np.newaxis]


    x = np.load("IIR_lowpass_filter.npy")
    a = x[0: n, 0]; a=a[:, np.newaxis]
    b = x[n: nm, 0]; b=b[:, np.newaxis]


    # Plot amplitude response
    rr = np.linspace(0, math.pi, num=Ns+1); rr=rr[:, np.newaxis]
    aa = np.vstack((np.ones((1, 1)) , a))
    AR = np.absolute(sig.freqz(b, aa, rr))
    plt.subplot(2, 3, 1)
    plt.plot(rr / math.pi, AR[1])
    plt.axis([0, 1, 0, 1.1])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude response')
    plt.title('IIR lowpass filter')


    # Plot group delay
    GD = sig.group_delay((b[:, 0], aa[:, 0]))
    rr = np.linspace(0, math.pi, num=512); rr=rr[:, np.newaxis]
    plt.subplot(2, 3, 2)
    plt.plot(rr / math.pi, GD[1])
    plt.axis([0, 1, 0, 20])
    plt.xlabel('Normalized frequency')
    plt.ylabel('Group delay')
    plt.title('IIR lowpass filter')


    # Plot port-zeros
    plt.subplot(2, 3, 3)
    tfx = control.tf(b[:, 0], aa[: , 0])
    control.pzmap(tfx)
    plt.axis([-2, 2, -2, 2])
    plt.grid()
    plt.title('Port-Zero diagram')


    # Signal simulation
    nnn = np.arange(0, 201); nnn=nnn[:, np.newaxis]
    w1 = 0.3 * math.pi
    w2 = 0.7 * math.pi
    x = np.cos(w1 * nnn) + np.sin(w2 * nnn)
    plt.subplot(2, 3, 4)
    plt.plot(nnn, x)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('Input x[n]')


    y = sig.lfilter(b[:, 0], aa[0,:], x[:, 0])
    plt.subplot(2, 3, 5)
    plt.plot(nnn, y)
    plt.axis([0, 200, -2, 2])
    plt.xlabel('n')
    plt.ylabel('Output y[n]')


    plt.show()






if __name__ == "__main__":
    main()