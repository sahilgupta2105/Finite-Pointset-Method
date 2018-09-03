import numpy as np
from numpy.linalg import inv

def MLS(dum_cloud,dum_idx,h):
    
    points_updated=dum_cloud
    point=points_updated[dum_idx]
    dum_nbd=point.get_nbd()
    # storing x and y coordinates of nbd
    nbd_x_temp=[]
    nbd_y_temp=[]
    for idx in dum_nbd:
        nbd_x_temp.extend([points_updated[idx].get_x()])
        nbd_y_temp.extend([points_updated[idx].get_y()])
    m=len(nbd_x_temp) # size of nbd list
    nbd_x=np.array(nbd_x_temp)
    nbd_y=np.array(nbd_y_temp)
    thisX=np.full(np.shape(nbd_x),point.get_x())
    thisY=np.full(np.shape(nbd_y),point.get_y())
    # constructing 'W' and 'M' matrices
    M= np.zeros((m,6))
    W= np.zeros((m,m))
    bU= np.zeros((m,))
    bV= np.zeros((m,))
    bP= np.zeros((m,))
    
    # filling 'b' matrix
    for dum_i in range(m):
        bU[dum_i]=points_updated[dum_nbd[dum_i]].get_u()
        bV[dum_i]=points_updated[dum_nbd[dum_i]].get_v()
        bP[dum_i]=points_updated[dum_nbd[dum_i]].get_p()            
    
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
    matU=np.matmul(mat3,bU)
    matV=np.matmul(mat3,bV)
    matP=np.matmul(mat3,bP)
    #print "Estimates (u,v,p): " + str(matU[0])+", "+str(matV[0])+", "+str(matP[0])
    points_updated[dum_idx].set_u(matU[0])
    points_updated[dum_idx].set_v(matV[0])
    points_updated[dum_idx].set_p(matP[0])
    
    return points_updated