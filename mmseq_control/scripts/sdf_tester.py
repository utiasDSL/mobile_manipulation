import numpy as np
import rospy
import time
import casadi as cs
from mobile_manipulation_central.ros_interface import MapInterface
from mmseq_control.map import SDF2D, SDF3D
from cbf_mpc.barrier_function2 import CBF, CBFJacobian

def test_sdf2d():
    tsdf_map_interface = MapInterface(topic_name="/pocd_slam_node/occupied_ef_dist_nodes")
    map = SDF2D()
    rate = rospy.Rate(20)

    while not tsdf_map_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        is_map_updated, tsdf = tsdf_map_interface.get_map()
        if is_map_updated:
            map.update_map(*tsdf)
        n_item = 1


        x = np.random.rand(n_item)*5
        y = np.random.rand(n_item)*5
        j = np.random.rand(n_item)
        x = [1.]
        y = [-0.5]
        input = np.concatenate((x,y))

        print(x,y)
        print('------Expected------')
        
        print(map.query_val(x, y))
        print(map.query_grad(x, y))
        

        print('------Casadi------')
        
        xs = cs.MX.sym('x',2*n_item)
        ys = cs.MX.sym('y',n_item)
        vs = cs.MX.sym('v',n_item)

        cbf = CBF("cbf", map)
        res = cbf(xs)
        print('res:',res)
        print('res eval:',cbf(input))

        print(cbf.jacobian())
        # jac = cbf.jacobian()(xs,res)
        jac = cs.jacobian(res, xs)
        #jac = cs.jacobian(res,xs,ys)
        print('jac:',jac.shape)
        cbf_grad = cs.Function('J_cbf',[xs],[jac])
        res_grad = cbf_grad(xs)
        print('grad',cbf_grad)

        eva_grad = cbf_grad(input)
        print('grad eval:',eva_grad)

        _ = np.random.rand(n_item)
        cbf_grad2 = CBFJacobian('H', map)
        res_grad2 = cbf_grad2(xs,res)
        print(cbf_grad2.jacobian())
        hess = cbf_grad2.jacobian()(xs,_,_)
        print(cbf_grad.jacobian())
        hess = cbf_grad.jacobian()(xs, jac)

        cbf_hess = cs.Function('H_cbf',[xs],[hess])
        eva_hess = cbf_hess(input)
        
        print('hess',eva_hess)


        time.sleep(0.1)

def test_sdf3d():
    tsdf_map_interface = MapInterface(topic_name="/pocd_slam_node/occupied_ef_dist_nodes")
    map = SDF3D()
    rate = rospy.Rate(20)

    while not tsdf_map_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        is_map_updated, tsdf = tsdf_map_interface.get_map()
        if is_map_updated:
            map.update_map(*tsdf)
        n_item = 1


        x = np.random.rand(n_item)*5
        y = np.random.rand(n_item)*5
        z = np.random.rand(n_item)*5
        j = np.random.rand(n_item)
        x = [1.]
        y = [-0.5]
        z = [0.4]
        input = np.concatenate((x,y,z))

        print(x,y,z)
        print('------Expected------')
        
        print(map.query_val(x, y, z))
        print(map.query_grad(x, y, z))
        

        print('------Casadi------')
        
        xs = cs.MX.sym('x',3*n_item)

        cbf = CBF("cbf", map, n=3)
        res = cbf(xs)
        print('res:',res)
        print('res eval:',cbf(input))

        print(cbf.jacobian())
        # jac = cbf.jacobian()(xs,res)
        jac = cs.jacobian(res, xs)
        #jac = cs.jacobian(res,xs,ys)
        print('jac:',jac.shape)
        cbf_grad = cs.Function('J_cbf',[xs],[jac])
        print('grad',cbf_grad)
        eva_grad = cbf_grad(input)
        print('grad eval:',eva_grad)
        
        _ = np.random.rand(n_item)
        cbf_grad2 = CBFJacobian('H', map, n=3)
        res_grad2 = cbf_grad2(xs,res)
        print(cbf_grad2.jacobian())
        hess = cbf_grad2.jacobian()(xs,_,_)
        print(cbf_grad.jacobian())
        hess = cbf_grad.jacobian()(xs, jac)

        cbf_hess = cs.Function('H_cbf',[xs],[hess])
        eva_hess = cbf_hess(input)
        
        print('hess',eva_hess)


        time.sleep(0.1)


if __name__ == '__main__':
    rospy.init_node('sdf_tester')
    test_sdf2d()
    test_sdf3d()