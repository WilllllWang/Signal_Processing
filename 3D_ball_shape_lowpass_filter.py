# Design of a 3D ball shape lowpass filter

import math
import numpy as np
from scipy import signal as sig
from scipy import fft 
from matplotlib import pyplot as plt



def main():
    # Independent parameters
    N = 15
    Wp = 0.55 * math.pi
    Ws = 0.75 * math.pi
    Ns = 25

    # Dependent parameters
    NH = (N - 1) // 2
    deltaW = math.pi / Ns
    NV = np.arange(0, NH + 1); NV=NV[: , np.newaxis]
    NH1 = NH + 1
    NH2 = NH1**2
    NH3 = NH1**3

    P = np.zeros((NH3, 1))
    Qp = np.zeros((NH3, NH3)) 
    Qs = np.zeros((NH3, NH3)) 
    Ns_p = 0
    Ns_s = 0

    # Reshape from 3D to 1D for simplicity
    for iw1 in range(0, Ns + 1):
        w1 = iw1 * deltaW
        for iw2 in range(0, Ns + 1):
            w2 = iw2 * deltaW
            for iw3 in range(0, Ns + 1):
                w3 = iw3 * deltaW
                cw = np.zeros((NH3, 1))
                for ii in range(0, NH + 1):
                    for i in range(0, NH + 1):
                        ss = ii * NH2 + i * NH1
                        ff = ii * NH2 + (i + 1) * NH1
                        cw[ss: ff, 0] = np.cos(ii * w1) * np.cos(i * w2) * np.cos(w3 * NV[:, 0])
                
                if (w1**2 + w2**2 + w3**2)**0.5 <= Wp:
                    Ns_p += 1
                    P -= 2 * cw
                    Qp += cw @ np.transpose(cw)
                elif (w1**2 + w2**2 + w3**2)**0.5 >= Ws:
                    Ns_s += 1
                    Qs += cw @ np.transpose(cw) 

    P *=  (1/8) * ((4/3) * math.pi * (Wp**3)) * (1/Ns_p)  # Volume of a sphere, 1/8
    Qp *= (1/8) * ((4/3) * math.pi * (Wp**3)) * (1/Ns_p)
    Qs *= (math.pi ** 3 - (1/8) * ((4/3) * math.pi * (Ws**3))) * (1/Ns_s)

    Q = Qp + Qs
    A = -0.5 * np.linalg.inv(Q) @ P

    # Reshape from 1D back to 3D
    A3 = np.reshape(A, (NH1, NH1, NH1))

    # Slicing W3 axis(fix W3) into 2D(total 3D) to show, since 4D is not possible
    fig = plt.figure(figsize = (6, 6))
    for ii in range(0, 6):
        w3 = ii * 0.2 * math.pi
        A2 = np.zeros((NH1, NH1))
        for in1 in range(0, NH + 1):
            for in2 in range(0, NH + 1):
                for in3 in range(0, NH + 1):
                    A2[in1, in2] += A3[in1, in2, in3] * np.cos(in3 * w3)

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
        
        ax = fig.add_subplot(2, 3, ii + 1, projection = '3d')
        ax.plot_surface(XX, YY, FR)
        plt.axis([-1, 1, -1, 1, 0, 1.1])
        plt.xlabel('w1')
        plt.ylabel('w2')

    plt.show()







if __name__ == "__main__":
    main()