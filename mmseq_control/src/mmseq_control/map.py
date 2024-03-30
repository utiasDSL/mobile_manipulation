import numpy as np
import threading
import rospy
from casadi import MX, DM
from visualization_msgs.msg import MarkerArray
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 


class SDF2D:
    def __init__(self, init_pose=np.eye(4)):
        self.mutex = threading.Lock()
        self.map = None
        self.valid = True
        self.init_robot_pose = init_pose
        self.inv_init_robot_pose = np.linalg.inv(init_pose)

        self.offset = 0.           #safety offset. should be kept the same as line 248 in ptcloud_vis.h
        self.mul = 10.0
        #self.default_val = 1.8 / 10.0 * self.mul - self.offset
        self.default_val = 1.8

        print('[map] init robot pose',self.init_robot_pose)

    
    def get_len(self,var):
        if (type(var) == DM):
            return DM.size(var)[0]
        elif (type(var) == np.float64):
            return 1 
        else:
            return len(var)

    def update_map(self, tsdf):
        if (len(tsdf) > 10):
            self.mutex.acquire(blocking=True)

            self.create_map(tsdf)
            self.valid = True

            self.mutex.release()

        
    def create_map(self, tsdf):
        pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
        #vs = [p.z*self.mul-self.offset for p in tsdf]
        vs = [p.z*self.mul for p in tsdf]
        self.map = LinearNDInterpolator(pts, vs) # choose LinearNDInterpolator(pts, vs) or CloughTocher2DInterpolator(pts, vs)

    def vis(self, tsdf):
        pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
        xs = pts[:,0]
        ys = pts[:,1]

        Nx = int(1.0/0.1 * max(xs-min(xs)))+1
        Ny = int(1.0/0.1 * max(ys-min(ys)))+1

        x_1d = np.linspace(min(xs), max(xs), Nx)
        y_1d = np.linspace(min(ys), max(ys), Ny)

        X,Y=np.meshgrid(x_1d, y_1d)
        Z=np.zeros(X.shape)
        # This makes the unobserved area free space
        for i in range(len(x_1d)):
            for j in range(len(y_1d)):
                Z[j][i] = self.query_val(x_1d[i], y_1d[j])
        #print(Z)

           
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X,Y,Z)
        plt.show()


    def query_val(self, x, y):
        self.mutex.acquire(blocking=True)
        val = self.query_val_nonblocking(x,y)
        self.mutex.release()
        return val
    
    def query_val_nonblocking(self, x, y):
        assert(self.get_len(x) == self.get_len(y))
        val = np.repeat(self.default_val, self.get_len(x))
        if (self.valid):
            # might need these lines when the robot does not start at the origin (have bugs)
            queried_pos = np.vstack((x,y,np.zeros(self.get_len(x)),np.ones(self.get_len(x))))
            transformed_pos = self.inv_init_robot_pose @ queried_pos
            new_x = transformed_pos[0,:]
            new_y = transformed_pos[1,:]
            #print('before',x,y)
            #print('after',new_x,new_y)
            val = self.map(new_x,new_y)
            #print("before",val)
            val = np.nan_to_num(val, True, self.default_val)
            #val[val == 0] = self.default_val
            #print("after",val)
        return val
    
    def query_grad(self, x, y):
        assert(self.get_len(x) == self.get_len(y))
        if not self.valid:
            return np.repeat([[0.0, 0.0]], self.get_len(x), axis=0)
        
        x = np.array(x)
        y = np.array(y)
        delta = 0.01
        self.mutex.acquire(blocking=True)

        val = self.query_val_nonblocking(x, y)
        val_pdx = self.query_val_nonblocking(x+delta, y)
        val_pdy = self.query_val_nonblocking(x, y+delta)
        val_mdx = self.query_val_nonblocking(x-delta, y)
        val_mdy = self.query_val_nonblocking(x, y-delta)

        #if val!=self.default_val and val_pdx!=self.default_val and val_mdx!=self.default_val:
        grad_x = (val_pdx - val_mdx) / (2*delta)
        #else:
        #    grad_x = 0.0

        #if val!=self.default_val and val_pdy!=self.default_val and val_mdy!=self.default_val:
        grad_y = (val_pdy - val_mdy) / (2*delta)
        #else:
        #    grad_y = 0.0
        self.mutex.release()
        return np.vstack((grad_x, grad_y))
