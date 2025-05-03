##
## Design of varable fractional delay (VFD) digital filters
##
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

N=50
M=7
wp=0.9*math.pi
Nw=200
Np=60
##
##
NH=N//2
deltaw=wp/Nw
deltap=1/Np
Mc=M//2
Ms=math.ceil(M/2)
nma=Mc*(NH+1)
nmb=Ms*NH
NV=np.arange(0,NH+1); NV=NV[:,np.newaxis]
NV1=np.arange(1,NH+1); NV1=NV1[:,np.newaxis]
Nwp=(Nw+1)*(Np+1)
##
##
ra=np.zeros((nma,1))
Qa=np.zeros((nma,nma))
for iw in range (0,Nw+1):
    w=iw*deltaw
    for ip in range (0,Np+1):
        p=-0.5+ip*deltap
        cwp=np.zeros((nma,1))
        for im in range (1,Mc+1):
            cwp[(im-1)*(NH+1):im*(NH+1),0]=p**(2*im)*np.cos(NV[:,0]*w)
        ra=ra-2*(np.cos(p*w)-1)*cwp
        Qa=Qa+cwp*np.transpose(cwp)
ra=wp*ra/Nwp
Qa=wp*Qa/Nwp
a=(-0.5*np.linalg.inv(Qa))@ra
##
##
rb=np.zeros((nmb,1))
Qb=np.zeros((nmb,nmb))
for iw in range (0,Nw+1):
    w=iw*deltaw
    for ip in range (0,Np+1):
        p=-0.5+ip*deltap
        swp=np.zeros((nmb,1))
        for im in range (1,Ms+1):
            swp[(im-1)*NH:im*NH,0]=p**(2*im-1)*np.sin(NV1[:,0]*w)
        rb=rb+2*np.sin(p*w)*swp
        Qb=Qb+swp*np.transpose(swp)
rb=wp*rb/Nwp
Qb=wp*Qb/Nwp
b=(-0.5*np.linalg.inv(Qb))@rb
##
##
a2=np.reshape(a,(Mc,NH+1)); a2=np.transpose(a2)
b2=np.reshape(b,(Ms,NH)); b2=np.transpose(b2)
##
##
h2=np.zeros((N+1,M+1))
##
## m=0
##
h2[NH,0]=1
##
## m:even
##
for im in range (1,Mc+1):
    h2[NH,2*im]=a2[0,im-1]
    h2[0:NH,2*im]=0.5*np.flipud(a2[1:NH+1,im-1])
    h2[NH+1:,2*im]=0.5*a2[1:,im-1]
##
## m:odd
##
for im in range (1,Ms+1):
    h2[0:NH,2*im-1]=0.5*np.flipud(b2[:,im-1])
    h2[NH+1:,2*im-1]=-0.5*b2[:,im-1]
##
##
MR=np.zeros((Nw+1,Np+1,1))
GD=np.zeros((512,Np+1))
h=np.zeros((N+1,1))
for ip in range (0,Np+1):
    p=-0.5+ip*deltap
    h=h2[:,0]
    for im in range (1,M+1): h=h+(p**im)*h2[:,im]
    rr=np.linspace(0,wp,num=Nw+1); rr=rr[:,np.newaxis]
    AR=np.absolute(signal.freqz(h,1,rr))
    MR[:,ip]=AR[1]

    GDD=signal.group_delay((h,1))
    GD[:,ip]=GDD[1]

plt.subplot(1,2,1)
for ip in range (0,Np+1): plt.plot(rr/math.pi,MR[:,ip])
plt.axis([0,wp/math.pi,0,1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude response')
plt.title('VFD')

plt.subplot(1,2,2)
rr=np.linspace(0,math.pi,num=512); rr=rr[:,np.newaxis]
for ip in range (0,Np+1): plt.plot(rr/math.pi,GD[:,ip])
plt.axis([0,wp/math.pi,NH-1,NH+1])
plt.xlabel('Normalized frequency')
plt.ylabel('Group-delay')
plt.title('VFD')  


plt.show()















##N=70
##wp1=0.08*math.pi
##wp2=0.12*math.pi
##wp3=0.685*math.pi
##wp4=0.715*math.pi
##Ns=10000
####
####
##deltaw=math.pi/Ns
##NV=np.arange(0,N+1); NV=NV[:,np.newaxis]
####
####
##PHD=np.zeros((Ns+1,1))
##PRD=np.zeros((Ns+1,1))
##W=np.zeros((Ns+1, 1))
##for iw in range (0,Ns+1):
##    w=iw*deltaw
##    if w <= wp1:
##        PHD[iw,0]=-85*w
##        W[iw,0]=90
##    elif w >= wp2 and w <= wp3:
##        PHD[iw,0]=-65*w-2*math.pi
##        W[iw,0]=1
##    elif w >= wp4:
##        PHD[iw,0]=-75*w+5*math.pi
##        W[iw,0]=30
##
##    PRD[iw,0]=-0.5*(N*w+PHD[iw,0])
####
####
##Q1=np.zeros((N+1,N+1))
##Q2=np.zeros((N+1,N+1))
##Q3=np.zeros((N+1,N+1))
##s1=0
##s2=0
##s3=0
##for iw in range (0,Ns+1):
##    w=iw*deltaw
##    C=np.sin(PRD[iw,0])*np.cos(NV[:,0]*w)+np.cos(PRD[iw,0])*np.sin(NV[:,0]*w)
##    C=C[:,np.newaxis]
##
##    if w <= wp1:
##        s1=s1+1
##        Q1=Q1+W[iw,0]*C@np.transpose(C)
##    elif w >= wp2 and w <= wp3:
##        s2=s2+1
##        Q2=Q2+W[iw,0]*C@np.transpose(C)
##    elif w >= wp4:
##        s3=s3+1
##        Q3=Q3+W[iw,0]*C@np.transpose(C)
##        
##Q1=wp1*Q1/s1
##Q2=(wp3-wp2)*Q2/s2
##Q3=(math.pi-wp4)*Q3/s3
##Q=Q1+Q2+Q3
####
####
##[D,X]=np.linalg.eig(Q)
##D=np.real(D)
##index=np.argmin(D)
##A=np.real(X[:,index]); A=A[:,np.newaxis]
####
####
##GD=signal.group_delay((np.flipud(A[:,0]),A[:,0]))
##rr=np.linspace(0,math.pi,num=512); rr = rr[:,np.newaxis]
##plt.subplot(1,2,1)
##plt.plot(rr/math.pi,GD[1])
##plt.axis([0,1,60,90])
##plt.xlabel('Normalized frequency')
##plt.ylabel('Group-delay') 
##plt.title('IIR allpass filter')
####
####
##plt.subplot(1,2,2)
##tfx=control.tf(np.flipud(A[:,0]),A[:,0])
##control.pzmap(tfx)
##plt.axis([-2,2,-2,2])
##plt.grid()
##plt.title('Pole-zero diagram')
##
##plt.show()

