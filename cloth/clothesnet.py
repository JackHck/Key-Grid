import numpy as np
import open3d as o3d
import os

path ='./fold/for_chengkai_top_1'
coat=[]
label = []
for i in range(0,2560,20):
    mesh= o3d.io.read_triangle_mesh(path+'/056_fix_timestep_' +str(i)+'.obj')
    vertices = np.asarray(mesh.vertices)
    #pcd = mesh.sample_points_poisson_disk(number_of_points=2048, init_factor=10) 
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    #o3d.io.write_point_cloud(path+'/'+str(file)+'/'+str(k)+'.pcd', pcd)
    #pcd_load = o3d.io.read_point_cloud(path+'/'+str(file)+'/'+str(k)+'.pcd')
    num_points = 128
    sampled_points = point_cloud.farthest_point_down_sample(num_samples=num_points)
    xyz_load = np.asarray(sampled_points.points)
    #xyz_load = np.asarray(pcd_load.points)
    coat.append(xyz_load)
    sampled_vertices = xyz_load
    indices = []
    for i in range(sampled_vertices.shape[0]):
        vertex = sampled_vertices[i]
        index = np.where((vertices == vertex).all(axis=1))[0][0]
        indices.append(index)
        label.append(np.array(indices))
        
coat= np.array(coat)
dmin = coat.min(axis=1, keepdims=True).min(axis=-1, keepdims=True)
dmax = coat.max(axis=1, keepdims=True).max(axis=-1, keepdims=True)
coat = (coat - dmin) / (dmax - dmin)
coat = 2.0 * (coat - 0.5)
coat = coat.reshape(-1, np.array(coat).shape[-1])
 
np.savetxt(path+'/'+"data_128.txt", coat) 
