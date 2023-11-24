import os
import open3d
import open3d as o3d
import seaborn as sns
import numpy as np
import pyvista as pv
import random


def save_kp_and_pc_in_pcd(pc, kp, output_dir, save=True, name=""):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''

    palette_PC = sns.color_palette()
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")

    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008) ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    ''' Add Keypoitnts '''
    for i in range(0, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.085) # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i==7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point

    if save:
        if not os.path.exists(output_dir+'/ply'):
            os.makedirs(output_dir+'/ply')
        o3d.io.write_triangle_mesh("{}/{}.ply".format(output_dir+'/ply', name), pcd)
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        cloud = pv.read("{}/{}.ply".format(output_dir+'/ply', name))
        colors = cloud.point_data['RGB'] / 255.0 
        cloud.point_data['RGB'] = colors
        
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        
        plotter.add_mesh(cloud, rgb=True)     
        plotter.camera_position = [1, -1, -1]
        camera = plotter.camera
        camera.SetViewUp(0,1,0)
        plotter.screenshot(r"{}/{}.png".format(output_dir+'/png', name))
        plotter.view_xy()
    else:
        o3d.visualization.draw_geometries([pcd])
    
