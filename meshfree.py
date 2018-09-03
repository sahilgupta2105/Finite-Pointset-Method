import numpy as np
from numpy.linalg import inv

def derivatives(dum_cloud,f,h):
    """
    computes the derivatives of the function 'f' using moving least squares
    """
    sol_mat=np.zeros((len(dum_cloud),5))
    dum_var=0
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
        M= np.zeros((m,6))
        W= np.zeros((m,m))
        b= np.zeros((m,))
        
        # filling 'b' matrix
        for dum_i in range(m):
            b[dum_i]=f[dum_nbd[dum_i]]            
        
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
        mat4=np.matmul(mat3,b)
        
        #filling the 'sol_mat'
        sol_mat[dum_var,0]=mat4[1]
        sol_mat[dum_var,1]=mat4[2]
        sol_mat[dum_var,2]=mat4[3]
        sol_mat[dum_var,3]=mat4[4]
        sol_mat[dum_var,4]=mat4[5]
        dum_var+=1
    
    return sol_mat