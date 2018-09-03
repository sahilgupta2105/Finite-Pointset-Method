import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import gmres
from scipy import sparse
from scipy import io as write_matlab
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import time
from scipy.sparse.csgraph import reverse_cuthill_mckee

def poisson_solver(dum_cloud,dum_rhs,h):
    
    A=np.zeros((len(dum_cloud),len(dum_cloud)))
    r=np.zeros((len(dum_cloud),))
    dum_var1=0
    f_rhs=dum_rhs
    
    k=0
    for point in dum_cloud:
        
        # indices of nbd of point
        dum_nbd= point.get_nbd()
        # storing x and y coordinates of nbd
        nbd_x_temp=[]
        nbd_y_temp=[]
        for idx in dum_nbd:
            nbd_x_temp.extend([dum_cloud[idx].get_x()])
            nbd_y_temp.extend([dum_cloud[idx].get_y()])
        m=len(nbd_x_temp) # size of nbd list
        nbd_x=np.array(nbd_x_temp)
        nbd_y=np.array(nbd_y_temp)
        thisX=np.full(np.shape(nbd_x),point.get_x())
        thisY=np.full(np.shape(nbd_y),point.get_y())
        # constructing 'W' and 'M' matrices
        if (thisY[0]>=1 and thisY[0]<1.1) or (thisY[0]<=0 and thisY[0]>-0.1) or (thisX[0]<=0 and thisX[0]>-0.1) or (thisX[0]>=1 and thisX[0]<1.1):
            M= np.zeros((m+2,6))
            W= np.zeros((m+2,m+2))
            
            # filling 'W' matrix
            W[m+1,m+1]=1.0
            W[m,m]=1.0
            
            # filling 'M' matrix
            # poisson's equation
            M[m,3]=1.0
            M[m,5]=1.0
            #Neumann boundary conditions
            if thisX[0]<=0 and thisX[0]>-0.1:
                M[m+1,1]=1
            elif thisX[0]>=1 and thisX[0]<1.1:
                M[m+1,1]=1
            elif thisY[0]<=0 and thisY[0]>-0.1:
                M[m+1,2]=1
            elif thisY[0]>=1 and thisY[0]<1.1:
                M[m+1,2]=1
            
            
        else:
            M= np.zeros((m+1,6))
            W= np.zeros((m+1,m+1))
            
            # filling 'W' matrix
            W[m,m]=1.0
            
            # filling 'M' matrix
                # poisson's equation
            M[m,3]=1.0
            M[m,5]=1.0
             
            
        
        # filling 'W' and 'M' elements
        deltaX=-thisX+nbd_x
        deltaY=-thisY+nbd_y
        delta=np.sqrt(np.square(deltaX)+np.square(deltaY))
        
        # 'W' and 'M' matrix
        for dum_i in range(m):
            if delta[dum_i]/h <=1.0:
                W[dum_i,dum_i]=np.exp(-6.25*(delta[dum_i]/h)**2)
            
            M[dum_i,0]=1
            M[dum_i,1]=deltaX[dum_i]
            M[dum_i,2]=deltaY[dum_i]
            M[dum_i,3]=0.5*deltaX[dum_i]**2
            M[dum_i,4]=deltaY[dum_i]*deltaX[dum_i]
            M[dum_i,5]=0.5*deltaY[dum_i]**2
        
        

        # finding the solution of optimization
        MTW=np.matmul(M.transpose(),W)
        mat1=np.matmul(MTW,M) # MTWM
        mat2=inv(mat1) # inverse of mat1
        mat3=np.matmul(mat2,MTW)    
        
        r[dum_var1]=f_rhs[dum_var1]*(mat3[0][m])
        for dum_i2 in range(m):
            A[dum_var1,dum_nbd[dum_i2]]= -1.0*mat3[0][dum_i2]
        A[dum_var1,dum_var1]=1.0
        dum_var1+=1
        
    sA = sparse.csr_matrix(A)        
    sol=gmres(sA,r,None,1e-5,30,100)
    dum_var2=0
    for point in dum_cloud:
        point.set_p(sol[0][dum_var2])
        dum_var2+=1
    
    #ax=Axes3D(plt.gcf())
    #ax.scatter(xx,yy,pp)

    return dum_cloud
        
