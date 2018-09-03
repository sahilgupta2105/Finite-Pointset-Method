import numpy as np
import matplotlib.pyplot as plt
from poisson_solver import poisson_solver
from mpl_toolkits.mplot3d import Axes3D
from meshfree import derivatives
from MLS import MLS
from scipy.spatial import Voronoi
from time import time
import random
from scipy import io as write_matlab

plt.axis([-0.2, 1.2, -0.2, 1.2])
plt.ion()

class voxel:
    """
    this implements the voxel data structure
    """
    def __init__(self,x,y):
        # lower left corner of square is stored
        self.x=x
        self.y=y
        self.points=[]
    
    def __str__(self):
        return "Voxel Cell (x,y,indicies): "+str(self.x)+", "+str(self.y)+", "+str(self.points)
    
    def clear_points(self):
        self.points=[]
        
    def add_point(self,idx):
        self.points.extend([idx])
        
    def get_points(self):
        return self.points
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y


class point_cloud:
    """
    this represents a point of the "Point Cloud"
    """
    def __init__(self,x,y,v_box,nbd_lst):
        self.x=x
        self.y=y
        self.u=0.0
        self.v=0.0
        self.p=0.0
        self.v_idx=v_box
        self.nbd=nbd_lst
        
    def __str__(self):
        return "Point info (x,y,voxel_idx,nbd): "+str(self.x)+", "+str(self.y)+", "+str(self.v_idx)+", "+str(self.nbd)
    
    def set_voxel(self,dum_v):
        self.v_idx=dum_v
        
    def set_nbd(self,nbd_lst):
        self.nbd=[]
        self.nbd.extend(nbd_lst)
        
    def get_voxel(self):
        return self.v_idx

    def get_nbd(self):
        return self.nbd
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

    def set_x(self,dum_x):
        self.x=dum_x
    
    def set_y(self,dum_y):
        self.y=dum_y

    def get_p(self):
        return self.p
    
    def set_p(self,dum_p):
        self.p=dum_p

    def get_u(self):
        return self.u

    def get_v(self):
        return self.v
    
    def set_u(self,dum_u):
        self.u=dum_u
        
    def set_v(self,dum_v):
        self.v=dum_v

# helper functions

def point_voxel_updater(dum_box,dum_cloud,dum_idx_lst=None):
    """
    updates the point to voxel and voxel to point info
    """
    #print dum_idx_lst
    flag=0
    if dum_idx_lst==None:
        dum_idx_lst=[]
        flag=1
        #print "len of dum_cloud: " + str(len(dum_cloud))
        for k in range(len(dum_cloud)):
            dum_idx_lst.extend([k])
            #k+=1
    if flag:
        # flush the voxel index points
        for vx in dum_box:
            vx.clear_points()
    
    for idx in dum_idx_lst:
        xIdx = dum_cloud[idx].get_x()
        yIdx = dum_cloud[idx].get_y()
        
        xVoxel = int((xIdx + 0.1)/h)
        yVoxel = int((yIdx + 0.1)/h)
        
        #print xVoxel,yVoxel
        idxVoxel = xVoxel + yVoxel*N
        
        dum_cloud[idx].set_voxel(idxVoxel)
        
        dum_box[idxVoxel].add_point(idx)
    
    return dum_box,dum_cloud

def nbd_search(dum_box,dum_cloud):
    """
    computes and stores the neighbours of each point in cloud
    """
    #    dum_debug=0
    for dum_pt in dum_cloud:
        voxel_idx=dum_pt.get_voxel()
        v_idx=get_nbd_voxel(voxel_idx)
        v_idx=list(set(v_idx))
    #        if dum_debug==1:
    #            print "DB: ",v_idx
        dum_nbd_lst=[]
        for dum_idx in v_idx:
            dum_nbd_lst.extend(dum_box[dum_idx].get_points())
        thisX=np.full(np.shape(dum_nbd_lst),dum_pt.x)
        thisY=np.full(np.shape(dum_nbd_lst),dum_pt.y)
        otherX=np.zeros(np.shape(dum_nbd_lst))
        otherY=np.zeros(np.shape(dum_nbd_lst))
        dum_var1=0
        for dum_idx1 in dum_nbd_lst:
            otherX[dum_var1]=dum_cloud[dum_idx1].x
            otherY[dum_var1]=dum_cloud[dum_idx1].y
            dum_var1+=1
        distances= np.sqrt(np.square(thisX-otherX)+np.square(thisY-otherY))
        nbd_lst1=np.where(np.logical_and(distances>0,distances<h))[0]
        nbd_lst2=[dum_nbd_lst[dum_i1] for dum_i1 in nbd_lst1]
        dum_pt.set_nbd(list(nbd_lst2))
    
    return dum_cloud
    #        dum_debug+=1

def plot_cloud(dum_cloud,it,dum_in=0):
    """
    plots the point cloud
    dum_cloud: point cloud data structure
    dum_in: plots variation of fluid property in domain
    """
    dum_lst1=[]
    dum_lst2=[]
    dum_lst3=[]
    dum_lst4=[]
    for dum_i5 in range(len(dum_cloud)):
        dum_lst1.extend([dum_cloud[dum_i5].x])
        dum_lst2.extend([dum_cloud[dum_i5].y])
        dum_lst3.extend([dum_cloud[dum_i5].get_u()])
        dum_lst4.extend([dum_cloud[dum_i5].get_v()])
    plt.clf()
    plt.plot(dum_lst1,dum_lst2,'.',color='b')
    plt.savefig('domain'+str(it)+'.png',dpi=700)
    plt.pause(0.05)
    plt.show()
    plt.clf()
    plt.quiver(dum_lst1,dum_lst2,dum_lst3,dum_lst4)
    plt.savefig('quiver'+str(it)+'.png',dpi=700)
    plt.pause(0.05)
    plt.show()
    plt.clf()
    plt.plot(err)
    plt.pause(0.05)
    plt.show()

def generate_cloud(dum_cloud):
    """
    generates a single point inside the domain randomly
    uses this as source to generate new points
    """
    points_updated=dum_cloud
    #points_updated=np.append(points_updated,point_cloud(random.uniform(0,1),random.uniform(0,1),0,[]))
    k=0
    dum_count=0
    print "Generating points"
    while k<4e10:
        #print "Random point generator: "+str(k)
        # Random point generator
        angle=(360)*np.random.rand(25,1)+0
        
        for i in range(np.prod(angle.shape)):
            pts_lst= cloud_list(points_updated)
            radius=(h-a*h)*random.uniform(0,1)+a*h
            # Get a trial point
            thisX= np.full(np.shape(points_updated),np.cos(angle[i]*np.pi/180)*radius+pts_lst[k,0])
            thisY= np.full(np.shape(points_updated),np.sin(angle[i]*np.pi/180)*radius+pts_lst[k,1])
            thisX=np.around(thisX,decimals=4)
            thisY=np.around(thisY,decimals=4)
            pts_x=np.array(pts_lst[:,0])
            pts_y=np.array(pts_lst[:,1])
            # Check distance form existing points
            distances= np.sqrt(np.square(thisX-pts_x)+np.square(thisY-pts_y))
            minDis= np.amin(distances)
            # Check if point meets the constraints
            if thisX[0]>1.1 or thisY[0]>1.1 or thisX[0]<-0.1 or thisY[0]<-0.1 or minDis<a*h:
                continue
            else:
                points_updated=np.append(points_updated,point_cloud(thisX[0],thisY[0],0,[]))

        dum_size=np.shape(points_updated)
        #plot_cloud(points_updated)
        # Condition is '-1' as array counts from zero
        if k-dum_size[0]==-1:
            if dum_count==1:
                break
            else:
                dum_count+=1
                k=0
        k=k+1
    print "Point generation completed! Generated "+str(len(points_updated))+" points!"
    return points_updated

def get_nbd_voxel(idx):
        v_idx1=idx-(N)-1
        v_idx2=idx-(N)
        v_idx3=idx-(N)+1
        v_idx4=idx-1
        v_idx5=idx
        v_idx6=idx+1
        v_idx7=idx+(N)-1
        v_idx8=idx+(N)
        v_idx9=idx+(N)+1
        if idx%N==0:
            v_idx_check=[v_idx2,v_idx3,v_idx5,v_idx6,v_idx8,v_idx9]
        elif idx%N==N-1:
            v_idx_check=[v_idx1,v_idx2,v_idx4,v_idx5,v_idx7,v_idx8]
        else:
            v_idx_check=[v_idx1,v_idx2,v_idx3,v_idx4,v_idx5,v_idx6,v_idx7,v_idx8,v_idx9]
        v_idx= [dum_i for dum_i in v_idx_check if dum_i>=0 and dum_i<(N)*(N)]
        return v_idx

def cloud_list(dum_cloud):
    """
    computes a numpy array of (x,y) of the input cloud
    """
    lst=np.zeros((len(dum_cloud),2))
    for dum_i in range(len(dum_cloud)):
        lst[dum_i][0]=dum_cloud[dum_i].x
        lst[dum_i][1]=dum_cloud[dum_i].y
        #print "DB: ",dum_cloud[dum_i].x,dum_cloud[dum_i].y
    return lst

def update_voxel_cloud(dum_box,dum_cloud,idx_lst=None):
    """
    updates the voxel boxes and point clouds using functions: point_voxel, voxel_point, nbd_search
    """
    [dum_box,dum_cloud]=point_voxel_updater(dum_box,dum_cloud,idx_lst)
    dum_cloud=nbd_search(dum_box,dum_cloud)
    
    return dum_box,dum_cloud

def boundary(dum_input1,dum_input2):
    """
    implements the VELOCITY boundary condition on input array
    dum_input1::: U VELOCITY
    dum_input2::: V VELOCITY
    """
    for dum_i in range(len(dum_input1)):
        tempX=points[dum_i].get_x()
        tempY=points[dum_i].get_y()
        if tempY<1.1 and tempY>=1:
            if tempX<0 or tempX>1:
                dum_input1[dum_i]=0
            else:
                dum_input1[dum_i]=16*((tempX)*(1-tempX))**2
            dum_input2[dum_i]=0.0
        elif tempY<=0 and tempY>-0.1:
            dum_input1[dum_i]=0.0
            dum_input2[dum_i]=0.0
        elif tempX<=0 and tempX>-0.1:
            dum_input1[dum_i]=0.0
            dum_input2[dum_i]=0.0
        elif tempX>=1 and tempX<1.1:
            dum_input1[dum_i]=0.0
            dum_input2[dum_i]=0.0
        else:
            pass

    return dum_input1,dum_input2

def cluster(dum_cloud):
    """
    recursive function detects clustered pair and adds new particle
    """
    points_updated=dum_cloud
    for point in points_updated:
        dum_nbd=point.get_nbd()
        nbdX=[]
        nbdY=[]
        for idx in dum_nbd:
            nbdX.extend([dum_cloud[idx].get_x()])
            nbdY.extend([dum_cloud[idx].get_y()])
        thisX=np.full((len(nbdX),),point.get_x())
        thisY=np.full((len(nbdX),),point.get_y())
        
        nbdX1=np.array(nbdX)
        nbdY1=np.array(nbdY)
        dis=np.sqrt(np.square(thisX-nbdX1)+np.square(thisY-nbdY1))
        dis_lst=np.where(np.logical_and(dis<=a*h,dis>0))[0]
        dis_lst1=[dum_nbd[idx] for idx in dis_lst]

        if dis_lst.size !=0:
            points_updated=np.delete(points_updated,dis_lst1)
            break
    
    return points_updated
    
def remove_outside_domain(dum_cloud):
    """
    removes points going outside domain
    """
    points_updated=dum_cloud
    dum_var=0
    for point in points_updated:
        thisX=point.get_x()
        thisY=point.get_y()
        if thisX> 1.1 or thisX< -0.1 or thisY> 1.1 or thisY< -0.1:
            points_updated=np.delete(points_updated,dum_var)
            break
        dum_var+=1
    
    return points_updated

def clean_domain(dum_cloud,dum_box):
    """
    cleans domain by invoking 'remove_outside_domain' recursively
    """
    points_updated=dum_cloud
    print "       Domain cleaning started!"
    while True:
        len1=len(points_updated)
        points_updated=remove_outside_domain(points_updated)
        len2=len(points_updated)
        [dum_box,points_updated]=update_voxel_cloud(dum_box,points_updated)
        if len1-len2==0:
            print "       Domain cleaning completed!"
            break
    
    return points_updated,dum_box

def add_remove_particles(dum_cloud,dum_box):
    """
    adds and removes clustered points in point cloud
    uses: 'cluster' and 'new_particles' functions
    """
    points_updated=dum_cloud
    print "Clustering algo started!"
    while True:
        # removing particles going outside domain
        #[points_updated,dum_box]= clean_domain(points_updated,dum_box)
        len1=len(points_updated)
        points_updated=cluster(points_updated)
        len2=len(points_updated)
        [dum_box,points_updated]=update_voxel_cloud(dum_box,points_updated)
        if len1-len2==0:
            print "Clustering completed!"
            break
        
        
    return points_updated,dum_box

def update_MLS(dum_cloud,idx_lst):
    """
    updates state variables at new points in domain using moving Least squares
    """
    points_updated=dum_cloud
    
    for idx in idx_lst:
        points_updated=MLS(points_updated,idx,h_W)
    
    return points_updated

def nbd_checker(dum_cloud):
    """
    checks if all points have enough neighbours
    """
    dum_lst = []
    for idx in range(len(dum_cloud)):
        if len(dum_cloud[idx].get_nbd()) < 7:
            dum_lst.extend([idx])
    print "Low neighbours at " + str(len(dum_lst)) + " locations"
    return dum_lst

def voronoi_hole_filler(dum_cloud, dum_box):
    points_updated = dum_cloud #these are all the neighbours of current voxel box
    #try to fill the upper boundary first
    upper_boundaryX = np.linspace(-0.1,1.1,int(1/(a*h))+1)
    upper_boundaryY = np.linspace(1.0,1.1,int(1/(a*h))+1)
    u_bc = np.ones((len(upper_boundaryX),2))
    for idx in range(len(upper_boundaryX)):
        u_bc[idx,0] = upper_boundaryX[idx]
        u_bc[idx,1] = upper_boundaryY[idx]
        
    pts_lst=cloud_list(points_updated)
    vor = Voronoi(pts_lst) #possible locations of all holes
    vor_vertices = vor.vertices
    initial_sz = len(points_updated)
    vor_vertices = np.concatenate((u_bc,vor_vertices))
    for vertex in vor_vertices:
        thisX= np.full(np.shape(points_updated),vertex[0])
        thisY= np.full(np.shape(points_updated),vertex[1])
        thisX=np.around(thisX,decimals=4)
        thisY=np.around(thisY,decimals=4)
        pts_x=np.array(pts_lst[:,0])
        pts_y=np.array(pts_lst[:,1])
        # Check distance form existing points
        distances= np.sqrt(np.square(thisX-pts_x)+np.square(thisY-pts_y))
        minDis= np.amin(distances)
        # Check if point meets the constraints
        if thisX[0]>1.1 or thisY[0]>1.1 or thisX[0]<-0.1 or thisY[0]<-0.1 or minDis<a*h:
            continue
        else:
            points_updated=np.append(points_updated,point_cloud(thisX[0],thisY[0],0,[]))
            pts_lst=cloud_list(points_updated)
            
    final_sz = len(points_updated)
    idx_lst=[]
    for i in range(initial_sz,final_sz):
        idx_lst.extend([i])
    [dum_box,points_updated]=update_voxel_cloud(dum_box,points_updated,idx_lst)
    points_updated=update_MLS(points_updated,idx_lst)
    nbd_checker(points_updated)
    return points_updated,dum_box
             
###############################_____MAIN PROGRAM STARTS HERE___#############################
# CONSTANTS
N_pts=21
h=0.13
a=0.25
h_W=h

########### POINT GENERATION BEGINS HERE################
# equispaced points for boundary
x=np.linspace(-0.1,1.1,N_pts)
y=np.linspace(-0.1,1.1,N_pts)

# hold the coordinates of all boundaries temporarily
b_lst_b=[]
b_lst_t=[]
b_lst_l=[]
b_lst_r=[]

for dum_i in range(N_pts):
    b_lst_b.append([x[dum_i],0.0])
    b_lst_t.append([x[dum_i],1.0])
    b_lst_l.append([0.0,y[dum_i]])
    b_lst_r.append([1.0,y[dum_i]])


 #creating boundary points in point cloud
points= np.empty((1,),dtype=object)
for dum_i1 in range(N_pts):
    pt_temp1=point_cloud(b_lst_b[dum_i1][0],b_lst_b[dum_i1][1],0,[])
    pt_temp2=point_cloud(b_lst_t[dum_i1][0],b_lst_t[dum_i1][1],0,[])
    pt_temp3=point_cloud(b_lst_l[dum_i1][0],b_lst_l[dum_i1][1],0,[])
    pt_temp4=point_cloud(b_lst_r[dum_i1][0],b_lst_r[dum_i1][1],0,[])
    points=np.append(points,pt_temp1)
    points=np.append(points,pt_temp2)
    points=np.append(points,pt_temp3)
    points=np.append(points,pt_temp4)
points=np.delete(points,0)
points=np.delete(points,0)
points=generate_cloud(points)

########## POINT GENERATION ENDS HERE################

############## VOXEL DATA STRUCTURE INITIALIZATION #################
N=int(1.2/h)+ (1.2%h>0)
voxel_box= np.empty(((N)*(N),),dtype=object)
N_shift=h*(N-1)
voxel_x=np.linspace(-0.1,N_shift,N)
voxel_y=np.linspace(-0.1,N_shift,N)
[voxel_xx,voxel_yy]= np.meshgrid(voxel_x,voxel_y)
voxel_xx=np.asarray(voxel_xx).reshape(-1)
voxel_yy=np.asarray(voxel_yy).reshape(-1)

# initialize voxel cells

for dum_itr in range((N)*(N)):
        voxel_box[dum_itr]=voxel(voxel_xx[dum_itr],voxel_yy[dum_itr])

############ VOXEL DATA STRUCTURE INITIALIZATION ENDS ######################

[voxel_box,points]=update_voxel_cloud(voxel_box,points)

############## TIME LOOP STARTS HERE #################
NT=300
DT=0.01
RE=100
RHO=100
L=1.0
U_INF=1.0
NU=U_INF*L/RE
err=[]
old_avg=0
solver_info = np.empty((300,6))

program_start = time()

for it in range(NT):

    print "Time: " + str(it*DT) + " and CPU time: " + str(time()-program_start)
    
    print "Domain size is " + str(len(points)) + " points"
        
    ####### Chorin's 1st STEP #######
    for dum_i2 in range(len(points)):
        tempX=points[dum_i2].get_x()+DT*points[dum_i2].get_u()
        tempY=points[dum_i2].get_y()+DT*points[dum_i2].get_v()
        points[dum_i2].set_x(tempX)
        points[dum_i2].set_y(tempY)
    
    tt1= time()
    # point cloud management
    [points,voxel_box]=clean_domain(points,voxel_box)
    [points,voxel_box]=add_remove_particles(points,voxel_box)
    [points,voxel_box]=voronoi_hole_filler(points,voxel_box)
    nbd_checker(points)
    tt2=time()
    solver_info[it][0]=len(points)
    solver_info[it][1]=tt2-tt1
    
    ### SPATIAL DERIVATIVES OF VELOCITY COMPONENTS ###
    tempU=[]
    tempV=[]
    for dum_i3 in range(len(points)):
        tempU.extend([points[dum_i3].get_u()])
        tempV.extend([points[dum_i3].get_v()])

    tt1=time()
    dU=derivatives(points,tempU,h_W)
    dV=derivatives(points,tempV,h_W)
    tt2=time()
    solver_info[it][2]=tt2-tt1

    ###### Chorin's 2nd STEP #######
    intU=np.zeros((len(points),))
    intV=np.zeros((len(points),))
    for dum_i4 in range(len(points)):
        intU[dum_i4]=points[dum_i4].get_u()+NU*DT*(dU[dum_i4,2]+dU[dum_i4,4])
        intV[dum_i4]=points[dum_i4].get_v()+NU*DT*(dV[dum_i4,2]+dV[dum_i4,4])

        # BOUNDARY CONDITION
    #[intU,intV]=boundary(intU,intV)
    tt1=time()
    int_dU=derivatives(points,intU,h_W)
    int_dV=derivatives(points,intV,h_W)
    tt2=time()
    solver_info[it][3]=tt2-tt1

    ###### Chorin's 3rd STEP #######
      # R.H.S. of Poisson's Equation
    dum_rhs=np.zeros((len(points),))
    for dum_i5 in range(len(points)):
        dum_rhs[dum_i5]=(int_dU[dum_i5,0]+int_dV[dum_i5,1])/DT

      # POISSON'S SOLVER
    start = time()
    points=poisson_solver(points,dum_rhs,h)
    stop = time()
    solver_info[it][4]=stop-start
    print "Poisson solver took " + str(stop-start) + "secs."
    tempP=[]
    for dum_i6 in range(len(points)):
        tempP.extend([points[dum_i6].get_p()])
    
    tt1=time()
    dP=derivatives(points,tempP,h_W)
    tt2=time()
    solver_info[it][5]=tt2-tt1
    ####### Chorin's 4th step #######
    temp_newU=[]
    temp_newV=[]
    for dum_i7 in range(len(points)):
        temp_U1=intU[dum_i7]-(DT)*(dP[dum_i7,0])
        temp_V1=intV[dum_i7]-(DT)*(dP[dum_i7,1])
        temp_newU.extend([temp_U1])
        temp_newV.extend([temp_V1])

      #BOUNDARY CONDITION
    [temp_newU,temp_newV]=boundary(temp_newU,temp_newV)
    
    current_avg=0
    for dum_i8 in range(len(points)):
        points[dum_i8].set_u(temp_newU[dum_i8])
        points[dum_i8].set_v(temp_newV[dum_i8])
        current_avg += np.sqrt(temp_newU[dum_i8]**2 + temp_newU[dum_i8]**2)
    current_avg = current_avg/len(points)
    err.extend([current_avg-old_avg])
    old_avg=current_avg    
    plot_cloud(points,it)

program_end = time()

print "Total run time: " + str(program_end-program_start)

xx=[]
yy=[]
uu=[]
vv=[]
pp=[]

for point in points:
    xx.extend([point.get_x()])
    yy.extend([point.get_y()])
    uu.extend([point.get_u()])
    vv.extend([point.get_v()])
    pp.extend([point.get_p()])

dic = {'xx':xx,'yy':yy,'uu':uu,'vv':vv,'pp':pp,'h':h,'a':a,'err':err,'solver_info':solver_info}

write_matlab.savemat('data.mat',dic)

while True:
    plt.pause(0.05)
