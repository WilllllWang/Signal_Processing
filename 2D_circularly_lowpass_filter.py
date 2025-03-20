# Design of a 2D circularly lowpass/highpass filter

import math
import numpy as np
import cv2 as cv
from scipy import signal as sig
from scipy import fft 
from matplotlib import pyplot as plt



def main():
    # Lowpass
    N = 21
    Wp = 0.05 * math.pi 
    Ws = 0.25 * math.pi
    Ns = 50

    NH = (N - 1) // 2
    deltaW = math.pi / Ns
    NV = np.arange(0, NH + 1); NV=NV[: , np.newaxis]
    NH1 = NH + 1
    NH2 = NH1 ** 2

    P = np.zeros((NH2, 1))
    Qp = np.zeros((NH2, NH2)) # Passband
    Qs = np.zeros((NH2, NH2)) # Stopband
    Ns_p = 0
    Ns_s = 0

    # Passband
    # Reshape from 2D to 1D for simplicity
    for iw1 in range(0, Ns + 1):
        w1 = iw1 * deltaW
        for iw2 in range(0, Ns + 1):
            w2 = iw2 * deltaW
            cw = np.zeros((NH2, 1))
            for i in range(0, NH + 1):
                cw[i * NH1: (i + 1) * NH1, 0] = np.cos(i * w1) * np.cos(w2 * NV[:, 0])
            
            if math.sqrt(w1 ** 2 + w2 ** 2) <= Wp:
                Ns_p += 1
                P -= 2 * cw
                Qp += cw @ np.transpose(cw)
            elif math.sqrt(w1 ** 2 + w2 ** 2) >= Ws:
                Ns_s += 1
                Qs += cw @ np.transpose(cw) 

    P *= 0.25 * (Wp ** 2) * math.pi / Ns_p
    Qp *= 0.25 * (Wp ** 2) * math.pi / Ns_p
    Qs *= (math.pi ** 2 - 0.25 * (Ws ** 2) * math.pi) / Ns_s

    Q = Qp + Qs
    A = -0.5 * np.linalg.inv(Q) @ P

    # Reshape from 1D back to 2D
    A2 = np.reshape(A, (NH1, NH1))


    ##---------------------------##
    # Convert A to h
    h = np.zeros((N, N))
    h[NH, NH] = A2[0, 0]
    
    h[0: NH, NH] = 0.5 * A2[NH: 0: -1, 0]
    h[NH+1: N, NH] = 0.5 * A2[1: NH+1, 0]
    
    h[NH, 0: NH] = 0.5 * A2[0, NH: 0: -1]
    h[NH, NH+1: N] = 0.5 * A2[0, 1: NH+1]

    h[0: NH, 0: NH] = 0.25 * A2[NH: 0: -1, NH: 0: -1]
    h[NH+1: N, 0: NH] = 0.25 * A2[1: NH+1, NH: 0: -1]
    h[0: NH, NH+1: N] = 0.25 * A2[NH: 0: -1, 1: NH+1]
    h[NH+1: N, NH+1: N] = 0.25 * A2[1: NH+1, 1: NH+1]


    # Plot input response
    FR = np.absolute(fft.fftshift(fft.fft2(h, (101, 101))))
    XX = np.zeros((101, 101))
    for i in range(0, 100 + 1): XX[:, i] = np.linspace(-1, 1, 101)
    YY = np.transpose(XX)
    
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(2, 3, 2, projection = '3d')
    ax.plot_surface(XX, YY, FR)
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('2D circularly lowpass filter')


    # Load image
    lena = cv.imread('lena.bmp')
    plt.subplot(2, 3, 4)
    plt.imshow(lena)
    
    # Filter the image
    lena_lowpass = cv.filter2D(src=lena, ddepth=-1, kernel=h)
    plt.subplot(2, 3, 5)
    plt.imshow(lena_lowpass)


    ##--------------------------------##
    ##--------------------------------##
    ##--------------------------------##
    # Highpass
    N = 21
    tmp = Ws
    Ws = Wp 
    Wp = tmp

    P = np.zeros((NH2, 1))
    Qp = np.zeros((NH2, NH2)) # Passband
    Qs = np.zeros((NH2, NH2)) # Stopband
    Ns_p = 0
    Ns_s = 0

    # Passband
    # Reshape from 2D to 1D for simplicity
    for iw1 in range(0, Ns + 1):
        w1 = iw1 * deltaW
        for iw2 in range(0, Ns + 1):
            w2 = iw2 * deltaW
            cw = np.zeros((NH2, 1))
            for i in range(0, NH + 1):
                cw[i * NH1: (i + 1) * NH1, 0] = np.cos(i * w1) * np.cos(w2 * NV[:, 0])
            
            if math.sqrt(w1 ** 2 + w2 ** 2) >= Wp:
                Ns_p += 1
                P -= 2 * cw
                Qp += cw @ np.transpose(cw)
            elif math.sqrt(w1 ** 2 + w2 ** 2) <= Ws:
                Ns_s += 1
                Qs += cw @ np.transpose(cw) 

    P *= (math.pi ** 2 - 0.25 * (Wp ** 2) * math.pi) / Ns_p
    Qp *= (math.pi ** 2 - 0.25 * (Wp ** 2) * math.pi) / Ns_p
    Qs *= 0.25 * (Ws ** 2) * math.pi / Ns_s

    Q = Qp + Qs
    A = -0.5 * np.linalg.inv(Q) @ P

    # Reshape from 1D back to 2D
    A2 = np.reshape(A, (NH1, NH1))


    ##---------------------------##
    # Convert A to h
    h = np.zeros((N, N))
    h[NH, NH] = A2[0, 0]
    
    h[0: NH, NH] = 0.5 * A2[NH: 0: -1, 0]
    h[NH+1: N, NH] = 0.5 * A2[1: NH+1, 0]
    
    h[NH, 0: NH] = 0.5 * A2[0, NH: 0: -1]
    h[NH, NH+1: N] = 0.5 * A2[0, 1: NH+1]

    h[0: NH, 0: NH] = 0.25 * A2[NH: 0: -1, NH: 0: -1]
    h[NH+1: N, 0: NH] = 0.25 * A2[1: NH+1, NH: 0: -1]
    h[0: NH, NH+1: N] = 0.25 * A2[NH: 0: -1, 1: NH+1]
    h[NH+1: N, NH+1: N] = 0.25 * A2[1: NH+1, 1: NH+1]


    # Plot input response
    FR = np.absolute(fft.fftshift(fft.fft2(h, (101, 101))))
    
    ax = fig.add_subplot(2, 3, 3, projection = '3d')
    ax.plot_surface(XX, YY, FR)
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('2D circularly highpass filter')

    
    # Filter the image
    lena_highpass = cv.filter2D(src=lena, ddepth=-1, kernel=h)
    plt.subplot(2, 3, 6)
    plt.imshow(lena_highpass)



    plt.show()









if __name__ == "__main__":
    main()