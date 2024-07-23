import numpy as np
import threading
import rospy
import time
import itertools
from casadi import MX, DM
import casadi as cs
from visualization_msgs.msg import MarkerArray
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator,RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from mmseq_utils import parsing

class SDF2D:
    def __init__(self, config, init_pose=np.eye(4)):
        self.mutex = threading.Lock()
        self.map = None
        self.valid = True
        self.dim = 2
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

    def update_map(self, tsdf, tsdf_vals):
        if (len(tsdf) > 10):
            self.mutex.acquire(blocking=True)

            self.create_map(tsdf, tsdf_vals)
            self.valid = True

            self.mutex.release()

    def set_robot_pose(self, pose):
        pass

    def get_params(self):
        return []
        
    def create_map(self, tsdf, tsdf_vals):
        pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
        #vs = [p.z*self.mul-self.offset for p in tsdf]
        # vs = [p.z*self.mul for p in tsdf]
        vs = [c.r * self.mul for c in tsdf_vals]

        self.map = LinearNDInterpolator(pts, vs) # choose LinearNDInterpolator(pts, vs) or CloughTocher2DInterpolator(pts, vs)
        self.map((0,0))
        
    def vis(self, x_lim, y_lim, block=True):
        Nx = int(1.0/0.1 * (x_lim[1] - x_lim[0]))+1
        Ny = int(1.0/0.1 * (y_lim[1] - y_lim[0]))+1

        x_1d = np.linspace(x_lim[0], x_lim[1], Nx)
        y_1d = np.linspace(y_lim[0], y_lim[1], Ny)

        X,Y=np.meshgrid(x_1d, y_1d)
        Z=np.zeros(X.shape)
        # This makes the unobserved area free space
        for i in range(len(x_1d)):
            for j in range(len(y_1d)):
                Z[j][i] = self.query_val(x_1d[i], y_1d[j])
        #print(Z)

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig, ax = plt.subplots()
        levels = np.linspace(-0.5, 1.5, int(2./0.25)+1)

        cs = ax.contour(X,Y,Z, levels)
        ax.clabel(cs, levels)
        ax.grid()
        ax.set_title("Signed Distance Field $sd(x)$")   # 0.6 is base collision radius
        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        plt.show(block=block)


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
        return np.hstack((grad_x, grad_y))

class SDF3D:
    def __init__(self, config, init_pose=np.eye(4)):
        self.mutex = threading.Lock()
        self.pose_mutex = threading.Lock()

        self.valid = False
        self.map = None
        self.dim = 3
        self.init_robot_pose = init_pose
        self.inv_init_robot_pose = np.linalg.inv(init_pose)
        self.curr_robot_pose = init_pose

        self.mul = 10.0
        self.default_val = 1.8

        self.voxel_size = 0.1
        self.map_size_xy = 4 # size of effective map in meters

        print('[map] init robot pose',self.init_robot_pose)

    def is_valid(self):
        return self.valid
    
    def get_len(self,var):
        if (type(var) == DM):
            return DM.size(var)[0]
        elif (type(var) == np.float64):
            return 1 
        else:
            return len(var)

    def update_map(self, tsdf, tsdf_vals):
        if (len(tsdf) > 10):
            self.create_map2(tsdf, tsdf_vals)
            self.valid = True
    
    def set_robot_pose(self, pose):
        self.pose_mutex.acquire(blocking=True)
        self.curr_robot_pose = pose
        self.pose_mutex.release()

    def get_params(self):
        return []
    
    def create_map0(self, tsdf, tsdf_vals):
        pts = np.around(np.array([np.array([p.x,p.y,p.z]) for p in tsdf]), 2).reshape((len(tsdf),3))
        vs = [c.r * self.mul for c in tsdf_vals]
        #print("max",max(vs))
        #print("min",min(vs))
        self.mutex.acquire(blocking=True)
        self.map = LinearNDInterpolator(pts, vs)
        self.map((0,0,0))

        self.mutex.release()
        
    def create_map(self, tsdf, tsdf_vals):
        pts = np.around(np.array([np.array([p.x,p.y,p.z]) for p in tsdf]), 2).reshape((len(tsdf),3))
        vs = [c.r * self.mul for c in tsdf_vals]
        #print("max",max(vs))
        #print("min",min(vs))
        self.map_ir = LinearNDInterpolator(pts, vs) # choose LinearNDInterpolator(pts, vs) or CloughTocher2DInterpolator(pts, vs) ort RegularGridInterpolator?
        time_first_query_start = time.time()
        self.map_ir(0,0,0)
        time_first_query_end = time.time()

        max_x = max(pts[:,0])
        min_x = min(pts[:,0])
        max_y = max(pts[:,1])
        min_y = min(pts[:,1])
        max_z = max(pts[:,2])
        min_z = min(pts[:,2])

        xs = np.linspace(min_x, max_x, int(np.ceil((max_x-min_x)/self.voxel_size)))
        ys = np.linspace(min_y, max_y, int(np.ceil((max_y-min_y)/self.voxel_size)))
        zs = np.linspace(min_z, max_z, int(np.ceil((max_z-min_z)/self.voxel_size)))
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
        time_data_start = time.time()
        data = np.nan_to_num(self.map_ir(xg, yg, zg), True, self.default_val)
        time_data_end = time.time()
        print("Time for first query",time_first_query_end-time_first_query_start)
        print("Time for data creation",time_data_end-time_data_start)

        self.mutex.acquire(blocking=True)
        #self.map = RegularGridInterpolator((xs, ys, zs), data, bounds_error=False, fill_value=self.default_val)
        self.map = self.map_ir
        self.map((0,0,0))
        self.mutex.release()

    def create_map2(self, tsdf, tsdf_vals):
        pts = np.around(np.array([np.array([p.x,p.y,p.z]) for p in tsdf]), 2).reshape((len(tsdf),3))
        vs = [c.r * self.mul for c in tsdf_vals]

        self.map_ir = LinearNDInterpolator(pts, vs) # choose LinearNDInterpolator(pts, vs) or CloughTocher2DInterpolator(pts, vs) ort RegularGridInterpolator?
        self.map_ir(0,0,0)

        # Limit the map to a certain size around the robot
        max_x = np.around(min(max(pts[:,0]), self.curr_robot_pose[0,3]+self.map_size_xy), 2)
        min_x = np.around(max(min(pts[:,0]), self.curr_robot_pose[0,3]-self.map_size_xy), 2)
        max_y = np.around(min(max(pts[:,1]), self.curr_robot_pose[1,3]+self.map_size_xy), 2)
        min_y = np.around(max(min(pts[:,1]), self.curr_robot_pose[1,3]-self.map_size_xy), 2)
        max_z = max(pts[:,2])
        min_z = min(pts[:,2])

        xs = np.linspace(min_x, max_x, int(np.ceil((max_x-min_x)/self.voxel_size)))
        ys = np.linspace(min_y, max_y, int(np.ceil((max_y-min_y)/self.voxel_size)))
        zs = np.linspace(min_z, max_z, int(np.ceil((max_z-min_z)/self.voxel_size)))

        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
        data = np.nan_to_num(self.map_ir(xg, yg, zg), True, self.default_val)

        self.mutex.acquire(blocking=True)
        #self.map = self.map_ir
        self.map = RegularGridInterpolator((xs, ys, zs), data, bounds_error=False, fill_value=self.default_val)
        self.map((0,0,0))
        self.mutex.release()
            
    # This assumes we only build the map around the filled up regions around the robot
    # In this region TSDF is assumed to form a regular grid
    def create_map3(self, tsdf, tsdf_vals):
        pts = np.around(np.array([np.array([p.x,p.y,p.z]) for p in tsdf]), 2).reshape((len(tsdf),3))
        vs = [c.r * self.mul for c in tsdf_vals]
        
        xs = np.unique(pts[:,0])
        ys = np.unique(pts[:,1])
        zs = np.unique(pts[:,2])

        val_dict = {}

        for idx in range(len(vs)):
            val_dict[(pts[idx,0],pts[idx,1],pts[idx,2])] = vs[idx]

        # must make sure the xy size is smaller than the filled up regions around the robot from mapping c++ code
        max_x = np.around(min(max(pts[:,0]), self.curr_robot_pose[0,3]+self.map_size_xy), 2)
        min_x = np.around(max(min(pts[:,0]), self.curr_robot_pose[0,3]-self.map_size_xy), 2)
        max_y = np.around(min(max(pts[:,1]), self.curr_robot_pose[1,3]+self.map_size_xy), 2)
        min_y = np.around(max(min(pts[:,1]), self.curr_robot_pose[1,3]-self.map_size_xy), 2)

        #remove the xyz pts outside the boundary
        xs = sorted(xs[(xs>=min_x) & (xs<=max_x)])
        ys = sorted(ys[(ys>=min_y) & (ys<=max_y)])

        data = np.ones((len(xs),len(ys),len(zs))) * self.default_val

        # Convert xs, ys, and zs to sets
        xs_set = set(xs)
        ys_set = set(ys)
        zs_set = set(zs)

        # Perform set intersection with the keys of val_dict
        keys = set(val_dict.keys()) & set(itertools.product(xs_set, ys_set, zs_set))
        #keys = set(itertools.product(xs_set, ys_set, zs_set))

        # Iterate over the keys and set the corresponding elements in the data array
        for (x, y, z) in keys:
            i = np.digitize(x, xs) - 1
            j = np.digitize(y, ys) - 1
            k = np.digitize(z, zs) - 1
            data[i, j, k] = val_dict[(x, y, z)]

        self.mutex.acquire(blocking=True)
        self.map = RegularGridInterpolator((xs, ys, zs), data, method="linear", bounds_error=False, fill_value=None) # extrapolate the values outside the map
        self.map((0,0,0))
        self.xs = xs
        self.ys = ys
        self.mutex.release()

    def vis(self, x_lim, y_lim, z_lim, block=True):
        Nx = int(1.0/0.1 * (x_lim[1] - x_lim[0]))+1
        Ny = int(1.0/0.1 * (y_lim[1] - y_lim[0]))+1
        Nz = int(1.0/0.1 * (z_lim[1] - z_lim[0]))+1

        dims = ['x', 'y', 'z']
        labels = ['x (m)', 'y (m)', 'z (m)']
        lims = np.vstack((x_lim, y_lim, z_lim))
        data_1d = []
        shown = []
        not_shown = 0
        for i, N in enumerate([Nx, Ny, Nz]):
            data_1d.append(np.linspace(lims[i][0], lims[i][1], N))
            if N!=1:
                shown.append(i)
            else:
                not_shown = i
        if len(shown)!=2:
            print(f"vis plot 2D data. {len(shown)}D data is given")

        x_idx = shown[0]
        y_idx = shown[1]
        X,Y=np.meshgrid(data_1d[x_idx], data_1d[y_idx])
        Z=np.zeros(X.shape)
        # This makes the unobserved area free space
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    val = self.query_val(data_1d[0][i], data_1d[1][j], data_1d[2][k])
                    temp = [i,j,k]
                    Z[temp[y_idx]][temp[x_idx]] = val
                    # print(f" {(data_1d[0][i], data_1d[1][j], data_1d[2][k])} = {val}")


        fig, ax = plt.subplots()
        levels = np.linspace(-1.0, 2.5, int(3.5/0.25)+1)

        cs = ax.contour(X,Y,Z, levels)
        ax.clabel(cs, levels)   
        ax.grid()
        ax.set_title(f"Signed Distance Field sd({dims[x_idx]}, {dims[y_idx]}, {data_1d[not_shown][0]})")   # 0.6 is base collision radius
        ax.set_xlabel(labels[x_idx])
        ax.set_ylabel(labels[y_idx])
        plt.show(block=block)


    def query_val(self, x, y, z):
        self.mutex.acquire(blocking=True)
        val = self.query_val_nonblocking(x,y,z)
        self.mutex.release()
        return val
    
    def query_val_nonblocking(self, x, y, z):
        assert(self.get_len(x) == self.get_len(y) == self.get_len(z))
        val = np.repeat(self.default_val, self.get_len(x))
        if (self.valid):
            # might need these lines when the robot does not start at the origin (have bugs)
            queried_pos = np.vstack((x,y,z,np.ones(self.get_len(x))))
            transformed_pos = self.inv_init_robot_pose @ queried_pos
            new_x = transformed_pos[0,:]
            new_y = transformed_pos[1,:]
            new_z = transformed_pos[2,:]

            val = self.map((new_x,new_y,new_z))
            # val = np.nan_to_num(val, True, self.default_val)

        return val
    
    def query_grad(self, x, y, z):
        assert(self.get_len(x) == self.get_len(y) == self.get_len(z))
        if not self.valid:
            return np.repeat([[0.0, 0.0, 0.0]], self.get_len(x), axis=0)
        
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        delta = 0.01
        self.mutex.acquire(blocking=True)

        val = self.query_val_nonblocking(x, y, z)
        val_pdx = self.query_val_nonblocking(x+delta, y, z)
        val_pdy = self.query_val_nonblocking(x, y+delta, z)
        val_pdz = self.query_val_nonblocking(x, y, z+delta)
        val_mdx = self.query_val_nonblocking(x-delta, y, z)
        val_mdy = self.query_val_nonblocking(x, y-delta, z)
        val_mdz = self.query_val_nonblocking(x, y, z-delta)

        #if val!=self.default_val and val_pdx!=self.default_val and val_mdx!=self.default_val:
        grad_x = (val_pdx - val_mdx) / (2*delta)
        #else:
        #    grad_x = 0.0

        #if val!=self.default_val and val_pdy!=self.default_val and val_mdy!=self.default_val:
        grad_y = (val_pdy - val_mdy) / (2*delta)
        #else:
        #    grad_y = 0.0
        grad_z = (val_pdz - val_mdz) / (2*delta)
        self.mutex.release()
        return np.hstack((grad_x, grad_y, grad_z))

class SDF2DNew:
    def __init__(self, config):

        self.dim = 2
        # These params should be the same as in MapInterface
        self.map_coverage = np.array(config["map"]["map_coverage"])
        self.default_val = config["map"]["default_val"]
        self.voxel_size = config["map"]["voxel_size"]
        self.map_size = np.ceil(self.map_coverage / self.voxel_size).astype(int)
        
        self.xg_sym = MX.sym("xg", self.map_size[0])
        self.yg_sym = MX.sym("yg", self.map_size[1])
        self.v_sym = MX.sym("v", self.map_size[0] * self.map_size[1])
        self.x_sym = MX.sym("x", self.dim)
        interp = cs.interpolant("interpol_linear_x", "linear", self.map_size)
        self.sdf_eqn = interp(self.x_sym, cs.vertcat(self.xg_sym, self.yg_sym), self.v_sym)
        self.sdf_fcn = cs.Function('map', [self.x_sym, self.xg_sym, self.yg_sym, self.v_sym], [self.sdf_eqn],
                                          ["x_query", "x_grid", "y_grid", "value"], ["sd(x)"])

        self.hess_eqn, self.grad_eqn = cs.hessian(self.sdf_eqn, self.x_sym)
        grad_eqn_normalized_list = [self.grad_eqn, self.grad_eqn/cs.norm_2(self.grad_eqn)]
        self.grad_eqn_normalized = cs.conditional(cs.norm_2(self.grad_eqn)>0.01,
                                                  grad_eqn_normalized_list, 0 , False)
        self.grad_fcn = cs.Function('map_grad', [self.x_sym, self.xg_sym, self.yg_sym, self.v_sym], [self.grad_eqn])
        self.grad_fcn_normalzied = cs.Function('map_grad_normalized', [self.x_sym, self.xg_sym, self.yg_sym, self.v_sym], [self.grad_eqn_normalized])


        self.xg, self.yg = self._get_default_grid()
        self.v = np.ones(self.map_size[0]* self.map_size[1]) * self.default_val

    def update_map(self, xg, yg, v):
        self.xg, self.yg, self.v = xg, yg, v
    
    def get_params(self):
        return [self.xg, self.yg, self.v]

    def vis(self, x_lim, y_lim, block=True):
        Nx = int(1.0/0.1 * (x_lim[1] - x_lim[0]))+1
        Ny = int(1.0/0.1 * (y_lim[1] - y_lim[0]))+1

        x_1d = np.linspace(x_lim[0], x_lim[1], Nx)
        y_1d = np.linspace(y_lim[0], y_lim[1], Ny)

        X,Y=np.meshgrid(x_1d, y_1d)
        Z=np.zeros(X.shape)
        dZdX = np.zeros((2, X.shape[0], X.shape[1]))
        # This makes the unobserved area free space
        for i in range(len(x_1d)):
            for j in range(len(y_1d)):
                Z[j][i] = self.query_val(x_1d[i], y_1d[j])
                dZdX[:, j, i] = self.query_grad(x_1d[i], y_1d[j]).flatten()
        #print(Z)

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig, ax = plt.subplots()
        levels = np.linspace(-0.5, self.default_val, int((self.default_val + 0.5)/0.25)+1)

        cs = ax.contour(X,Y,Z, levels)
        ax.clabel(cs, levels)
        ax.grid()
        ax.quiver(X, Y, dZdX[0],dZdX[1], scale_units="xy", scale=1, color='gray')

        ax.set_title("Signed Distance Field $sd(x)$")   # 0.6 is base collision radius
        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        plt.show(block=block)


    def query_val(self, x, y):
        val = self.sdf_fcn(np.vstack((x,y)), self.xg, self.yg, self.v)
        return val
    
    def query_grad(self, x, y):
        return self.grad_fcn(np.vstack((x,y)), self.xg, self.yg, self.v).toarray()
    
    def _get_default_grid(self):
        # Limit the map to a certain size around the robot
        xg = np.linspace(-self.map_coverage[0]/2, self.map_coverage[0]/2, self.map_size[0])
        yg = np.linspace(-self.map_coverage[1]/2, self.map_coverage[1]/2, self.map_size[1])

        return xg, yg

class SDF3DNew:
    def __init__(self, config):

        self.dim = 3
        self.default_val = config["map"]["default_val"]
        self.map_coverage = parsing.parse_array(config["map"]["map_coverage"]) # x,y in m
        self.voxel_size = config["map"]["voxel_size"]

        self.map_size = np.ceil(self.map_coverage / self.voxel_size).astype(int)
        self.mul=10

        self.xg_sym = MX.sym("xg", self.map_size[0])
        self.yg_sym = MX.sym("yg", self.map_size[1])
        self.zg_sym = MX.sym("zg", self.map_size[2])

        self.v_sym = MX.sym("v", self.map_size[0] * self.map_size[1]*self.map_size[2])
        self.x_sym = MX.sym("x", self.dim)
        interp = cs.interpolant("interpol_linear_x", "linear", self.map_size)
        self.sdf_eqn = interp(self.x_sym, cs.vertcat(self.xg_sym, self.yg_sym, self.zg_sym), self.v_sym)
        self.sdf_fcn = cs.Function('map', [self.x_sym, self.xg_sym, self.yg_sym, self.zg_sym, self.v_sym], [self.sdf_eqn],
                                          ["x_query", "x_grid", "y_grid", "z_grid", "value"], ["sd(x)"])

        self.hess_eqn, self.grad_eqn = cs.hessian(self.sdf_eqn, self.x_sym)
        self.grad_fcn = cs.Function('map_grad', [self.x_sym, self.xg_sym, self.yg_sym, self.zg_sym, self.v_sym], [self.grad_eqn])
        self.hess_fcn = cs.Function('map_hess', [self.x_sym, self.xg_sym, self.yg_sym, self.zg_sym, self.v_sym], [self.hess_eqn])
        self.xg, self.yg, self.zg = self._get_default_grid()
        self.v = np.ones(self.map_size[0]* self.map_size[1]*self.map_size[2]) * self.default_val

    def update_map(self, xg, yg, zg, v):
        self.xg, self.yg, self.zg, self.v = xg, yg, zg, v
    
    def get_params(self):
        return [self.xg, self.yg, self.zg, self.v]

    def vis(self, x_lim, y_lim, z_lim, block=True):
        Nx = int(1.0/0.1 * (x_lim[1] - x_lim[0]))+1
        Ny = int(1.0/0.1 * (y_lim[1] - y_lim[0]))+1
        Nz = int(1.0/0.1 * (z_lim[1] - z_lim[0]))+1

        dims = ['x', 'y', 'z']
        labels = ['x (m)', 'y (m)', 'z (m)']
        lims = np.vstack((x_lim, y_lim, z_lim))
        data_1d = []
        shown = []
        not_shown = 0
        for i, N in enumerate([Nx, Ny, Nz]):
            data_1d.append(np.linspace(lims[i][0], lims[i][1], N))
            if N!=1:
                shown.append(i)
            else:
                not_shown = i
        if len(shown)!=2:
            print(f"vis plot 2D data. {len(shown)}D data is given")

        x_idx = shown[0]
        y_idx = shown[1]
        X,Y=np.meshgrid(data_1d[x_idx], data_1d[y_idx])
        Z=np.zeros(X.shape)
        dZdX = np.zeros((3, X.shape[0], X.shape[1]))
        # This makes the unobserved area free space
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    val = self.query_val(data_1d[0][i], data_1d[1][j], data_1d[2][k])
                    temp = [i,j,k]
                    Z[temp[y_idx]][temp[x_idx]] = val
                    # print(f" {(data_1d[0][i], data_1d[1][j], data_1d[2][k])} = {val}")
                    dZdX[:, temp[y_idx], temp[x_idx]] = self.query_grad(data_1d[0][i], data_1d[1][j], data_1d[2][k]).flatten()


        fig, ax = plt.subplots()
        levels = np.linspace(-1.0, 2.5, int(3.5/0.25)+1)

        cs = ax.contour(X,Y,Z, levels)
        ax.clabel(cs, levels)   
        ax.quiver(X, Y, dZdX[x_idx],dZdX[y_idx], scale_units="xy", scale=1, color='gray')
        ax.grid()
        ax.set_title(f"Signed Distance Field sd({dims[x_idx]}, {dims[y_idx]}, {data_1d[not_shown][0]})")   # 0.6 is base collision radius
        ax.set_xlabel(labels[x_idx])
        ax.set_ylabel(labels[y_idx])
        plt.show(block=block)

    def query_val(self, x, y, z):
        input = np.vstack((x,y,z))
        val = self.sdf_fcn(input, self.xg, self.yg, self.zg, self.v)

        return val
    
    def query_grad(self, x, y, z):
        return self.grad_fcn(np.vstack((x,y,z)), self.xg, self.yg, self.zg, self.v).toarray()

    def query_hessian(self, x, y, z):
        return self.hess_fcn(np.vstack((x,y,z)), self.xg, self.yg, self.zg, self.v).toarray()

    def _get_default_grid(self):
        # Limit the map to a certain size around the robot

        max_x = np.around(self.map_coverage[0]/2, 2)
        min_x = np.around(-self.map_coverage[0]/2, 2)
        max_y = np.around(self.map_coverage[1]/2, 2)
        min_y = np.around(-self.map_coverage[1]/2, 2)
        min_z = 0
        max_z = self.map_coverage[2]

        xg = np.linspace(min_x, max_x, self.map_size[0])
        yg = np.linspace(min_y, max_y, self.map_size[1])
        zg = np.linspace(min_z, max_z, self.map_size[2])

        return xg, yg, zg
