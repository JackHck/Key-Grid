import os
import open3d
import open3d as o3d
import seaborn as sns
import numpy as np
import pyvista as pv
import random

def pc_to_pcd(pc):
    palette_PC = sns.color_palette()
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)  ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    return pcd


def kp_to_pcd(kp):
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
    pcd.translate(kp[0])
    pcd.paint_uniform_color(palette[0])

    for i in range(1, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)  # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i == 7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point
    return pcd


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
        
        '''
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        render_options = vis.get_render_option()
        render_options.point_size = 2  # 设置点的大小
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        vis.capture_screen_image("{}/{}.png".format(output_dir+'/png', name))
        #ctr = vis.get_view_control()
        #ctr.scale(50)  # 设置缩放系数
        #ctr.set_zoom(20)  # 设置视角
        vis.destroy_window()
        '''
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        cloud = pv.read("{}/{}.ply".format(output_dir+'/ply', name))
        colors = cloud.point_data['RGB'] / 255.0 
        # 创建一个Plotter对象并设置正交投影
        cloud.point_data['RGB'] = colors
        
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        #plotter.view_xy()
        # 添加点云到Plotter
        #se3 = [[1, 1, -1],[-1, 1, -1],[1, 1, -1],[1, -1, 1],[-1, -1, 1],[-1,-1,-1]]
        #winner = random.choice(se3)
        #plotter.camera_position = winner
        #plotter.camera_position = [1, -1, -1]
        #camera = plotter.camera
        #a = [-1,0,1]
        #b = [-1,0,1]
        #c = [-1,0,1]
        #a = random.choice(a)
        #b = random.choice(b)
        #c = random.choice(c)
        #camera.SetViewUp(a,b, c)
        #camera.SetViewUp(0,5,0)
        plotter.add_mesh(cloud, rgb=True)     
        plotter.camera_position = [1, -1, -1]
        camera = plotter.camera
        camera.SetViewUp(0,1, 0)
        plotter.screenshot(r"{}/{}.png".format(output_dir+'/png', name))
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        plotter.view_xy()
        # 添加点云到Plotter
    else:
        o3d.visualization.draw_geometries([pcd])
    
