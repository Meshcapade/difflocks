
#run with 
# ~/blender-4.1.1-linux-x64/blender  -t 8 --background --python ./inference/npz2blender.py -- --input_npz <NPZ_PATH> --out_path <OUTPUT_PATH> --export_alembic







import bpy
from bpy.app.handlers import persistent
import bpy_extras
import os
import numpy as np
# from gloss import *
# import trimesh
# import json
import argparse
import sys
# import imageio.v3 as iio
from os import listdir
from os.path import isfile, join

import math
from mathutils import Matrix, Vector
import mathutils
import shutil
import time



path_cur_script=os.path.dirname(os.path.abspath(__file__))

def export_alembic(out_alembic_path, resolution):
    print("-------------------------------------------")
    
    # bpy.ops.outliner.item_activate(deselect_all=True)
    bpy.data.objects["hair_01"].select_set(True)
    # hair = bpy.context.active_object
    hair = bpy.data.objects["hair_01"]
    bpy.context.view_layer.objects.active=hair


    start=time.time()
    for modif in hair.modifiers:
        print("applying",modif.name)
        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.modifier_apply(modifier=modif.name)
    print("finished applying all geometry nodes")
    end=time.time()
    print("applying geometry nodes took", end-start)
    #shrinkwrap on the scalp (Wrong because it makes weird strands for the long hair)
    #default hair with t=8: 20s
    #default hair with t=16: 15s
    #50% strans with t=16: 6s

    #with shrinkwrap on the whole mesh
    #default hair with t=8: 43s
    #default hair with t=16: 34s
    #50% strans with t=16: 14s
    #50% strans, 50%points with t=16: 7s
    #50% strans, 25%points with t=16: 5s




    #conver particle
    bpy.ops.curves.convert_to_particle_system()

    # bpy.ops.outliner.item_activate(deselect_all=True)
    # bpy.context.space_data.context = 'PARTICLES'
    bpy.context.object.show_instancer_for_render = False
    bpy.context.object.show_instancer_for_viewport = False
    #I have no idea which one actually works to increase resolution so I change all
    # bpy.data.particles["ParticleSettings"].display_step = 7
    # bpy.data.particles["ParticleSettings"].hair_step = 7
    # bpy.data.particles["ParticleSettings"].render_step = 7
    bpy.data.particles["ParticleSettings"].display_step = resolution
    bpy.data.particles["ParticleSettings"].hair_step = resolution
    bpy.data.particles["ParticleSettings"].render_step = resolution
    

    #hide everything except scalp
    for obj in bpy.data.objects:
        print("obj", obj)
        if obj.name!="smplx_scalp_blender":
            obj.hide_render=True
            obj.hide_viewport=True
        else:
            print("smplx scalp blender doesn't get hidden")
    # for obj in bpy.scene.objects:
        # print("obj in scene", obj)



    bpy.data.objects['smplx_scalp_blender'].hide_render=False
    bpy.data.objects['smplx_scalp_blender'].hide_viewport=False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_render = False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_viewport = False
    bpy.data.objects["smplx_scalp_blender"].select_set(True)



    bpy.ops.wm.alembic_export(filepath=out_alembic_path, check_existing=False, start=1, end=1,selected=True, visible_objects_only=True, uvs=False, packuv=False, normals=False, use_instancing=False, global_scale=1.0, export_hair=True, export_particles=False, as_background_job=False, evaluation_mode='VIEWPORT', init_scene_frame_range=True)



def main():
    print("main")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True) #npz file to read and create a alembic from
    parser.add_argument('--out_path', required=True) #output path for the blender file and the alembic
    parser.add_argument('--export_alembic', action='store_true') #set it to true to also export an alembic file
    parser.add_argument('-ss', '--strands_subsample', type=float, default=1.0)  # perentage of strands we keep (1.0=keep all, 0.5=keep half, 0.25=keep quarter)
    parser.add_argument('-vs', '--vertex_subsample', type=float, default=1.0)  # perentage of vertices per strand to keep (1.0=keep all, 0.5=keep half, 0.25=keep quarter)
    parser.add_argument('-ar', '--alembic_resolution', type=int, default=7) #the resolution of the alembic, higher number means more points per strand (default=7 which is probably 2^7=128 points per strands)
    parser.add_argument('-sh', '--shrinkwrap', action='store_true') #set it to true to perform a shrinkwrap of the hair so that it avoids penetrating through the body
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    print("strands_subsample", args.strands_subsample)
    print("vertex_subsample", args.vertex_subsample)
    print("alembic_resolution", args.alembic_resolution)
    print("shrinkwrap", args.shrinkwrap)


    #read npz 
    path_hair=args.input_npz
    do_export_alembic=args.export_alembic


    hair_geom=np.load(path_hair)
    points=hair_geom["positions"] #nr_strands x nr_points_per_strand x 3


    subsample_nr_strands=False
    #removes randomly x amount of strands or X nr of vertices
    if args.strands_subsample!=1.0 or args.vertex_subsample!=1.0:
        subsample_nr_strands=True
    if subsample_nr_strands:
        print("before ramoving random curves, points is ", points.shape) #nr_strands x nr_verts x3
        num_strands_to_keep = int(points.shape[0] * args.strands_subsample)
        strands_to_keep = np.random.choice(points.shape[0], num_strands_to_keep, replace=False)
        points = points[strands_to_keep, :, :].copy()
        print("after removing random curves, points is ", points.shape)

        #removing verts now 
        nr_verts_to_skip=int(np.floor(1.0/args.vertex_subsample))
        print("nr_verts_to_skip",nr_verts_to_skip)
        points = points[:, ::nr_verts_to_skip, :].copy()
        print("after removing consecurive vertices, points is ", points.shape)
    print("final points", points.shape)

    #open the blender file
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v24.blend")
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v25_with_shrinkwrap.blend")
    # if args.shrinkwrap:
    #     path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    # else:
    #     path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v24.blend")
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v27_blender36.blend")
    path_in_blend= "/localhome/aha220/Hairdar/assets/body_models/blender_scenes/smplx/vis_base_v26_shrinkwrap.blend"
    bpy.ops.wm.open_mainfile(filepath=path_in_blend)


    #write new geometry
    print("creating geometry")
    bpy.data.objects["hair_01"].select_set(True)
    obj = bpy.data.objects.get("hair_01")
    bpy.context.view_layer.objects.active = obj
    # bpy.ops.object.mode_set(mode='EDIT')
    # #  Get the evaluated state of the object to account for geometry nodes and modifiers
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # depsgraph.update()
    # eval_obj = obj.evaluated_get(depsgraph)
    # curves_data=eval_obj.data
    # help(obj.data)
    curves_data=obj.data
    nr_strands=points.shape[0]
    # nr_strands=3000
    nr_points_per_strand=points.shape[1]

    #v4 faster
    points_per_curve = [nr_points_per_strand for i in range(nr_strands)]
    curves_data.add_curves(points_per_curve)
    # print("added curves")
    # exit(1)

    # Prepare a flat array for positions
    flat_points = points.reshape(-1, 3)  # Flatten points to a 2D array
    flat_points[:, [1, 2]] = flat_points[:, [2, 1]]  # Swap y and z
    flat_points[:, 1] *= -1  # Negate the y values

    # Assign the flat array directly
    curves_data.points.foreach_set("position", flat_points.flatten())

    if not args.shrinkwrap:
        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")

    # Update the viewport to reflect changes
    obj.data.update_tag()
    obj.modifiers.update()
    # bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.context.view_layer.update()



    #save blend file
    print('saving .blend')
    out_scene_path=os.path.join(args.out_path, "blender_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_scene_path) 
    print('finished saving .blend')


    if do_export_alembic:
        out_path_alembic=os.path.join(args.out_path, "hair.abc")
        print("exporting hair to", out_path_alembic)
        export_alembic(out_path_alembic, args.alembic_resolution)

   
if __name__ == '__main__':
    main() 
