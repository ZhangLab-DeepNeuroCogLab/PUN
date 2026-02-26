import numpy as np
from tqdm import tqdm
import os
import sys
from PIL import Image
from pathlib import Path
from gtsam import Point3, Pose3, Rot3
import torch
import bpy
import bpycv
import mathutils
import trimesh
import math
# root_path = os.getenv('nbv_root_path', '/default/path')
# shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')
# distribution_dataset_path = os.getenv('distribution_dataset_path', '/default/distribution/dataset/path')
# sys.path.append(root_path)
import tyro
from config import *
from contextlib import contextmanager
import json
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")
# os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

from fep_nbv.env.utils import rgb_to_rgba, sharpness, load_from_json, write_to_json, pose_point_to,save_img
import fep_nbv.env.gen_data_fn as gen_data_fn
# import nvf.env.gen_data_fn as gen_data_fn
# from nvf.env.BlenderRenderer import *
# from nvf.env.BlenderProcess import BlenderProcess
# from nvf.env.utils import get_conf, pose_point_to, rgb_to_rgba, sharpness, variance_of_laplacian, load_from_json, write_to_json



def convert_to_blender_pose(pose, return_matrix=False):
    if type(pose) is torch.Tensor:
        rot = Rot3.Quaternion(pose[3],*pose[:3])
        pose = Pose3(rot, pose[4:]).matrix()
    if type(pose) is Pose3:
        pose = pose.matrix()
    pose = mathutils.Matrix(pose)
    if return_matrix:
        return pose
    else:
        return pose.to_translation(), pose.to_euler()

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

# from contextlib import contextmanager
# import os
# import sys

# @contextmanager
# def stdout_redirected(to=os.devnull):
#     """
#     Safer stdout redirection for environments where sys.stdout may not behave as expected.
#     """
#     if not hasattr(sys.stdout, "fileno"):
#         # If sys.stdout is not a file-like object, skip redirection
#         print("Warning: sys.stdout has no fileno attribute, skipping redirection.")
#         yield
#         return

#     try:
#         fd = sys.stdout.fileno()
#     except Exception as e:
#         print(f"Error accessing sys.stdout.fileno(): {e}")
#         yield
#         return

#     # Backup the original stdout descriptor
#     old_stdout_fd = os.dup(fd)

#     try:
#         # Redirect stdout to the target file
#         with open(to, 'w') as file:
#             os.dup2(file.fileno(), fd)
#             yield
#     finally:
#         # Restore the original stdout descriptor
#         os.dup2(old_stdout_fd, fd)
#         os.close(old_stdout_fd)

def extract_mesh_data(obj, all_vertices, all_faces, num_existing_vertices):
    # If the object is a mesh, extract its data
    if obj.type == 'MESH':
        # Access the mesh data of the object
        mesh = obj.data

        # Get the object's world matrix
        matrix_world = obj.matrix_world

        # Create a BMesh from the mesh data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        # Triangulate the mesh (convert non-triangle faces to triangles)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")

        # Iterate through the vertices of the BMesh
        for vertex in bm.verts:
            # Apply the object's world matrix to the vertex coordinates
            vertex_world = matrix_world @ vertex.co
            all_vertices.append(vertex_world)

        # Iterate through the faces of the BMesh and adjust vertex indices
        for face in bm.faces:
            # Convert local vertex indices to global ones
            face_verts = [v.index + num_existing_vertices for v in face.verts]
            all_faces.append(face_verts)

        # Update the number of existing vertices
        num_existing_vertices += len(bm.verts)

    # Recursively extract mesh data from children
    for child in obj.children:
        num_existing_vertices = extract_mesh_data(child, all_vertices, all_faces, num_existing_vertices)

    return num_existing_vertices

def trimesh_load_mesh(path):
    mesh_or_scene = trimesh.load(path)

    # if self.cfg.scale !=1.:
    #     matrix = np.eye(4)
    #     matrix[:2, :2] *= self.cfg.scale
    #     mesh_or_scene.apply_transform(matrix)

    if isinstance(mesh_or_scene, trimesh.Scene):
        # If the scene contains multiple meshes, you have a few options:
        
        # Option 1: Get a single mesh (if you know there's only one)
        if len(mesh_or_scene.geometry) == 1:
            mesh = next(iter(mesh_or_scene.geometry.values()))

        # Option 2: Combine all meshes into one
        else:
            mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        # The loaded object is already a Trimesh object
        mesh = mesh_or_scene
    return mesh


class BaseScene(object):
    name = ''
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.set_camera_params()

        self.gen_data_fn = {}
        self._mesh = None

        # self.cfg.scale
    def set_camera_params(self):
        self.height, self.width = self.cfg.resolution
        
        self.hfov = self.cfg.fov/180 *np.pi
        self.vfov = self.cfg.fov/180 *np.pi

        self.fx = 0.5*self.width/np.tan(self.hfov/2 )
        self.fy = 0.5*self.height/np.tan(self.vfov/2 )


    def set_camera_pose(self, wTc):
        raise NotImplementedError
    def render(self):
        raise NotImplementedError
    def get_mesh(self):
        raise NotImplementedError
    
    @property
    def intrinsic_matrix(self):
        return np.array([[self.fx, 0, self.width/2], [0, self.fy, self.height/2], [0, 0, 1]])

    @property
    def mesh(self):
        if self._mesh is None:
            print("Starts extracting mesh")
            self._mesh = self.get_mesh()
            print("Ends extracting mesh")
        return self._mesh
    def get_aabb(self):
        return np.array([self.mesh.vertices.min(axis=0), self.mesh.vertices.max(axis=0)])
    
    def render_pose(self, pose):
        
        self.set_camera_pose(pose)
        
        result = self.render() # 
        # img = result['mask']*result['image']
        img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
        return img
    
    def render_poses(self, poses):
        img_list = []
        for i,pose in enumerate(tqdm(poses)):
        
            self.set_camera_pose(pose)
            result = self.render()
            img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
            img_list.append(img)
        return img_list
    
    def get_camera_params(self):
        params = {"camera_angle_x": self.hfov,
            "camera_angle_y": self.hfov,
            "fl_x": self.fx,
            "fl_y": self.fy,
            "k1": 0.,
            "k2": 0.,
            "p1": 0.,
            "p2": 0.,
            "cx": self.width/2.,
            "cy": self.width/2.,
            "w": self.width,
            "h": self.width}
        return params
    
    def load_data(self, file, idx_start=None, idx_end=None, img_path=None, return_quat=True):
        """copies images from hubble dataset used for testing add_images()"""
        # path='cfg/initial_transforms.json'
        # hubble_dataset_path = "data/nerfstudio/hubble_mask/img"
        # hubble_dataset_path = "/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/img"
        # files = Path(hubble_dataset_path).glob('*')

        transforms_dict = load_from_json(Path(file))
        params = self.get_camera_params()
        # assert params == {k:transforms_dict[k] for k in params.keys()}
        for k in params.keys():
            assert np.allclose(params[k], transforms_dict[k]), f'{k} is not equal: {params}, {transforms_dict}'
        # transforms_dict = load_from_json(Path("/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/transforms/transforms.json"))
        # transforms_dict = load_from_json(Path("data/nerfstudio/hubble_mask/transforms.json"))

        np_images = []
        transforms = []
        i = 0
        if idx_start is None:
            idx_start=0
        if idx_end is None:
            idx_end=len(transforms_dict["frames"])

        for i in range(idx_start, idx_end):
            if img_path:
                image = Image.open(os.path.join(img_path,f"{i:04}.png"))
            else:
                image = Image.open(os.path.join(os.path.dirname(file),transforms_dict["frames"][i]['file_path']))
            np_image = np.array(image)
            # np_image = rgb_to_rgba(np_image)
            np_images.append(np_image)

            transform = np.asarray(transforms_dict["frames"][i]['transform_matrix'])
            # import pdb; pdb.set_trace()
            if return_quat:
                pose = Pose3(transform)  
                q = pose.rotation().toQuaternion() # w, x, y, z
                t = pose.translation()
                transforms.append(torch.FloatTensor([q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2]]))
            else:
                transforms.append(torch.FloatTensor(transform))

        return np_images, transforms
    
    def save_data(self, file, poses, images):
        assert file.endswith('.json')
        # assert len(poses[0].shape) > 1 # pose in matrix form
        output_path = os.path.dirname(file)
        output_path_img = os.path.join(output_path, "img")
        os.makedirs(output_path_img, exist_ok=True)

        frames = []
        for i,pose in enumerate(poses):
            img = images[i]
            img_path = os.path.join('img', f'{i:04d}.png')

            dd = {"file_path": img_path,
                "sharpness": sharpness(img),
                "transform_matrix": pose.tolist()
                }
            frames.append(dd)
            
            im = Image.fromarray(img)
            im.save(os.path.join(output_path, img_path))

        data = self.get_camera_params()
        data.update({
        "aabb_scale": 1,
        "frames": frames
        })
        write_to_json(Path(file), data)
    
class Blender(BaseScene):
    def __init__(self, cfg=None):
        super(Blender, self).__init__(cfg)
        
        self.scene = bpy.context.scene

        

        self.postprocess_fn = None

        self.valid_range = np.array([[-2.,-2.,-2.], [2.,2.,2.]])
        # self.worlds = {}
        # self.current_world = None
        self.camera_scale = 1.

    
    def config_camera(self, camera=None):
        if camera is None:
            camera = self.camera
        camera.data.type = 'PERSP'
        camera.data.sensor_fit = 'HORIZONTAL'
        camera.data.sensor_width = 36.0
        camera.data.sensor_height = 24.0

        # height,width = self.cfg.resolution
        # fx = 0.5*self.width/np.tan(self.cfg.hfov/2)
        # fy = 0.5*self.width/np.tan(self.cfg.hfov/2)
        camera.data.lens = self.fx / self.width * camera.data.sensor_width

        # pixel_aspect = self.fy / self.fx
        # scene = bpy.data.scenes["Scene"]
        # scene.render.pixel_aspect_x = 1.0
        # scene.render.pixel_aspect_y = pixel_aspect

        # camera.data.dof_object = focus_target
        # camera.data.cycles.aperture_type = 'RADIUS'
        # camera.data.cycles.aperture_size = 0.100
        # camera.data.cycles.aperture_blades = 6
        camera.data.dof.use_dof = False
        self.scene.camera = camera

    def config_blender(self):
        # set up rendering
        render_settings = self.scene.render
        render_settings.resolution_x = self.cfg.resolution[1]
        render_settings.resolution_y = self.cfg.resolution[0]
        render_settings.resolution_percentage = self.cfg.resolution_percentage
        render_settings.use_file_extension = True
        render_settings.image_settings.file_format = 'PNG'

        self.set_engine(cycles=True)

    def set_engine(self, cycles=True):
        if cycles and self.cfg.cycles:
            bpy.context.scene.render.engine = 'CYCLES'
            self.scene.cycles.samples = self.cfg.cycles_samples
            if self.cfg.gpu:
                bpy.context.preferences.addons[
                    "cycles"
                ].preferences.compute_device_type = "CUDA"
                bpy.context.scene.cycles.device = "GPU"
            else:
                bpy.context.scene.cycles.device = "CPU"


            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            self.current_engine = 'cycles'
        else:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            self.current_engine = 'eevee'

    # def set_world(self, world_name):
        
    #     if self.current_world == world_name:
    #         return
    #     self.scene.world = self.worlds[world_name]
    #     self.current_world = world_name

    def add_camera(self, camera_matrix=np.eye(4)):
        rot = mathutils.Matrix(camera_matrix).to_euler()
        translation = mathutils.Matrix(camera_matrix).to_translation()
        bpy.ops.object.camera_add(location=translation, rotation=rot)
        camera = bpy.context.object
        self.camera = camera
        return camera
    
    def set_camera_pose(self, wTc, camera=None):
        self.camera_pose = wTc
        if hasattr(self, 'camera'):
            if camera is None:
                camera = self.camera
            loc, rot = convert_to_blender_pose(wTc)
            camera.location = loc * self.camera_scale
            camera.rotation_euler = rot

    def render(self):
        with stdout_redirected():
            material_indices_backup = {}
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    mesh = obj.data
                    material_indices_backup[obj.name] = [poly.material_index for poly in mesh.polygons]
            bpy.context.view_layer.update()
            
            result = bpycv.render_data()

            # 恢复材质
            for obj_name, indices in material_indices_backup.items():
                obj = bpy.data.objects.get(obj_name)
                if obj and obj.type == 'MESH':
                    mesh = obj.data
                    for poly, material_index in zip(mesh.polygons, indices):
                        poly.material_index = material_index
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        # bpy.context.view_layer.update()
        rgb = np.array(result['image'])
        depth = np.array(result['depth'])
        inst = np.array(result['inst'])
        mask = np.where(inst>0, 1, 0).astype(bool)[...,None]
        return {'image': rgb, 'depth': depth, 'mask': mask}
    
    def set_mesh_filename(self, filepath):
        self.mesh_filepath=filepath

    def get_mesh_filename(self):
        if self.mesh_filepath is None:
            self.mesh_filepath = "data/assets/blend_files/room.obj"
        return self.mesh_filepath
    
    def get_mesh(self):
        if(self.cfg.scene=="RoomScene"):
            # Get all objects in the scene
            filename = self.get_mesh_filename()
            mesh = trimesh.load(filename,force='mesh')
            TF=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
            mesh = mesh.apply_transform(TF)
            bounding_box = trimesh.primitives.Box(bounds=self.valid_range)
            clipped_mesh = mesh.intersection(bounding_box)
            return mesh
        
        # Get all objects in the scene
        all_objects = bpy.context.scene.objects

        # Initialize lists to store all vertices and faces
        all_vertices = []
        all_faces = []
        # Initialize a variable to keep track of the number of existing vertices
        num_existing_vertices = 0

        # Iterate through all objects
        for obj in all_objects:
            if not obj.parent:
                num_existing_vertices = extract_mesh_data(obj, all_vertices, all_faces, num_existing_vertices)
        # breakpoint()
        all_vertices = np.array(all_vertices) / self.camera_scale
        all_faces = np.array(all_faces, dtype=np.int64)
        # breakpoint()
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

        if self.postprocess_fn:
            mesh = self.postprocess_fn(mesh)

        bounding_box = trimesh.primitives.Box(bounds=self.valid_range)

        clipped_mesh = mesh.intersection(bounding_box)

        return clipped_mesh

class BlenderFile(Blender):
    """docstring for Blender"""
    def __init__(self, cfg=None):
        super(BlenderFile, self).__init__(cfg)
    
    def obj_filter(self, obj):
        return False

    def load_scene(self, scene_path, scale=1.):
        self.camera_scale = 1/scale
        bpy.ops.wm.open_mainfile(filepath=scene_path)
        self.scene = bpy.context.scene
        # breakpoint()
        self.camera = bpy.context.scene.camera
        
        

        # scale = 1.
        # obj = bpy.context.selected_objects[0]
        # obj.scale = scale * np.ones(3)
        # print(obj.type)
        # input()
        # inst_id = len(self.objects) +1000
        inst_id = 1000
        # inst_id = 1
        # bpycv.material_utils.set_vertex_color_material(obj)
        # obj["inst_id"] = inst_id
        # print(bpy.context.active_object["inst_id"])
        # input()
        # breakpoint()
        for i, ob in enumerate(bpy.data.objects): 
            # if ob.parent == obj: 
            # ob["inst_id"]= inst_id+i+1
            # breakpoint()
            # if ob.name.startswith('Plane'):
            #     breakpoint()
            # loc = np.array(ob.location)
            # bounds = self.valid_range *10
            # if outofbounds(loc, bounds):
            #     breakpoint()
            #     ob['inst_id'] = 0
            #     print(ob.name, 'is out of range')
            if self.obj_filter(ob):
                ob["inst_id"]= 0
            else:
                ob["inst_id"]= 1
            # ob["inst_id"]= inst_id
        # with bpycv.activate_obj(obj):
        #     bpy.ops.rigidbody.object_add()
        # self.objects[key] = obj

        # self.camera = bpy.data.objects.get("Camera")
        self.add_camera()
        self.config_camera()
        self.config_blender()
    
    def add_camera(self, camera_matrix=np.eye(4)):

        # camera = bpy.data.objects.get("Camera")
        # # Deselect all objects
        # bpy.ops.object.empty_add(location=[0.,0.,0.])
        # empty = bpy.context.active_object

        # bpy.context.view_layer.objects.active = camera
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # # 4. Clear the camera's parent
        # camera.parent = None
        
        # # Optionally, delete the empty
        # bpy.data.objects.remove(empty)

        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        # 2. Add a new camera to the scene
        return super().add_camera(camera_matrix)
    
    def render(self):
        return super().render()

class ShapeNetBlender(BlenderFile):
    def __init__(self,cfg=None):
        super(ShapeNetBlender, self).__init__(cfg)
        self.objects = {}

        self.hdri_rotation = [0.0, 0.0, 0.0] # wRi
        self.hdri_path = None

    def init_rgb_world(self, name='rgb'):
        world = bpy.data.worlds.new(name=name)
        world.use_nodes = True
        # self.worlds[name] = world
        node_tree = world.node_tree
        # node_tree = bpy.data.node_groups.new(type="ShaderNodeTree", name="RGBNodeTree")
        # background_node = node_tree.nodes.new(type='ShaderNodeBackground')

        environment_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        if self.hdri_path:
            environment_texture_node.image = bpy.data.images.load(self.hdri_path)
        self.lighting_texture_node = environment_texture_node

        mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
        self.lighting_mapping_node = mapping_node
        mapping_node.inputs["Rotation"].default_value = tuple(self.hdri_rotation)

        tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")

        node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        node_tree.links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
        node_tree.links.new(environment_texture_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])
        self.rgb_node_tree= node_tree

        # 添加点光源
        # for obj in bpy.data.objects:
        #     if obj.type == 'LIGHT':
        #         bpy.data.objects.remove(obj, do_unlink=True)
        # light_data = bpy.data.lights.new(name="MyLight", type='POINT')
        # light_data.energy = 1000
        # light_object = bpy.data.objects.new(name="MyLight", object_data=light_data)
        # bpy.context.collection.objects.link(light_object)
        # light_object.location = (2.0, -2.0, 3.0)
    
    def init_albedo_world(self, name='albedo'):
        pass
    def init_mask_world(self, name='mask'):
        pass

    def init_depth_world(self, name='depth'):
        pass
    
    def set_lighting(self, rotation=None, hdri_path=None):
        if rotation:
            if type(rotation) is Rot3:
                rotation = rotation.ypr()[::-1].tolist() # TODO: need double check
            self.lighting_mapping_node.inputs["Rotation"].default_value = tuple(rotation)
            self.hdri_rotation = tuple(rotation)
        if hdri_path:
            self.lighting_texture_node.image = bpy.data.images.load(hdri_path)
            self.hdri_path = hdri_path

    def add_object(self, key, obj_file_path, obj_matrix=np.eye(4), scale=1.):
        # json_path = obj_file_path[:-4]+'.json'
        # with open(json_path, 'r') as file:
        #     data = json.load(file)
        #     self.centroid = data.get("centroid", None)
        with stdout_redirected():
            bpy.ops.import_scene.obj(filepath=obj_file_path)
        obj = bpy.context.selected_objects[0]
        # obj.location = mathutils.Vector(self.centroid)
        # print(f'target location: {obj.location}')
         # 获取对象的包围盒顶点（转换为世界坐标）
        # world_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        # min_corner = mathutils.Vector(map(min, zip(*world_corners)))
        # max_corner = mathutils.Vector(map(max, zip(*world_corners)))
        # print(f"World Min: {min_corner}, World Max: {max_corner}")
        
        obj.scale = scale * np.ones(3)
        # if scale != 1.:
        #     bpy.ops.transform.resize(value=(scale, scale, scale))
        # obj.location += mathutils.Vector((0,0,0.2))
        # print(obj.type)
        # input()
        inst_id = len(self.objects) +1000
        # inst_id = 1
        # bpycv.material_utils.set_vertex_color_material(obj)
        obj["inst_id"] = inst_id
        # print(bpy.context.active_object["inst_id"])
        # input()
        for i, ob in enumerate(bpy.data.objects): 
            if ob.parent == obj: 
                ob["inst_id"]= inst_id+i+1
                # ob["inst_id"]= inst_id
        with bpycv.activate_obj(obj):
            bpy.ops.rigidbody.object_add()
        self.objects[key] = obj

    def delete_object(self):
        # 对象名称
        object_name = "model_normalized"  # 替换为要删除的对象名称

        # 查找对象
        obj = bpy.data.objects.get(object_name)

        # 删除对象
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
            print(f"Deleted object: {object_name}")
        else:
            print(f"Object '{object_name}' not found.")
        
    def set_object_pose(self, key, wTo):
        obj = self.objects[key]
        loc, rot = convert_to_blender_pose(wTo)
        obj.location = loc
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = rot
        pass

class ShapeNetScene(ShapeNetBlender):
    """docstring for ShapeNetScene"""
    name = 'shapenet'
    def __init__(self, cfg):
        super(ShapeNetScene, self).__init__(cfg)
        self.hdri_rotation = [0.0, 0.0, 0.0] if not hasattr(cfg, 'hdri_rotation') else cfg.hdri_rotation # wRi
        # self.hdri_path = os.path.join(cfg.root,"assets/hdri/RenderCrate-HDRI_Orbital_38_4K.hdr")
        self.hdri_path = os.path.join(cfg.root,"assets/hdri/gray_hdri.exr")
        self.target_path = cfg.target_path

        # self.hdri_path = "assets/hdri/neon_photostudio_4k.hdr"
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
        self.add_camera()
        self.add_object('target', self.target_path, scale=self.cfg.scale)
        print(f'object scale by {self.cfg.scale}')
        self.config_camera()
        self.config_blender()
        self.init_rgb_world()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.shapenet_init(),
            'eval': lambda :gen_data_fn.shapenet_eval(), # radius=[self.cfg.radius]
            'full': lambda :gen_data_fn.shapenet_full(), # radius=[self.cfg.radius]
        }
        # self.set_world('rgb')

    def get_mesh(self):
        mesh = trimesh.load(self.target_path,force='mesh')
        return mesh 

    def set_white_background(self):
        # 设置世界背景颜色为白色
        bpy.context.scene.world.use_nodes = True
        world = bpy.context.scene.world
        bg_node = world.node_tree.nodes.get('Background')
        
        if bg_node:
            bg_node.inputs[0].default_value = (1, 1, 1, 1)  # RGBA白色背景
        
        # 如果使用的是渲染引擎为 Eevee，确保将背景设置为透明
        bpy.context.scene.render.film_transparent = False
    
    def add_cone_at_pose(self, position, quaternion,id=0):
        # 将quaternion从xyzw的顺序转换为wxyz的顺序
        # quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        # 如果id不等于0，先将前一个圆锥的颜色设置为黑色
        if id != 0:
            cone = self.objects.get(f'Cone_{id-1}')
            if cone:
                mat = cone.data.materials[0]
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
                bsdf.inputs['Alpha'].default_value = 0.2
        

        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.5, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)  # 红色
        bsdf.inputs['Roughness'].default_value = 0  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        bsdf.inputs['Specular'].default_value = 4.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()
        
        # 旋转圆锥，使底面朝向给定的方向
        # rotation_quaternion = mathutils.Quaternion(quaternion)
        # rotation_quaternion = mathutils.Quaternion([0.7071, 0, 0.7071, 0])
        # cone.rotation_quaternion = rotation_quaternion

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone

    def add_cone_at_pose2(self, position, quaternion,id=0, color=(1, 0, 0, 1)):
        '''
        - input
            position xyz
            quaternion wxyz
        '''
        
        # 创建一个圆锥作为视角指示器
        bpy.ops.mesh.primitive_cone_add(radius1=0.1, depth=0.3, location=(0, 0, 0))
        cone = bpy.context.object
        cone.name = f'ViewCone_{id}'
            
        # 添加红色材质
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True  # 使用节点系统
        # mat.diffuse_color = (1, 0, 0, 1)  # 红色
        # 获取节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 设置颜色和镜面反射
        bsdf.inputs['Base Color'].default_value = color  # 红色
        # bsdf.inputs['Roughness'].default_value = 1  # 控制镜面反射的粗糙度（0 为光滑，1 为粗糙）
        # bsdf.inputs['Specular'].default_value = 1.0  # 镜面反射强度
        cone.data.materials.append(mat)
        
        # 设置圆锥的位置
        cone.location = mathutils.Vector(position)
        import math
        cone.rotation_euler = mathutils.Quaternion(quaternion).to_euler()

        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cone["inst_id"] = inst_id
        self.objects[f'Cone_{id}'] = cone
        print(f'new cone id is {inst_id}')

        # if id>0:
        #     cone = self.objects.get(f'Cone_{id}')
        #     if cone:
        #         mat = cone.data.materials[0]
        #         bsdf = mat.node_tree.nodes.get("Principled BSDF")
        #         bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0.2)  # 黑色
        #         bsdf.inputs['Alpha'].default_value = 0.2


    def add_transparent_cube(self, aabb, cube_color=(1,0,0,0.5),edge_color=(1, 0, 0, 1), edge_width=0.5):
        min_corner = aabb[0, :].tolist()
        max_corner = aabb[1, :].tolist()

        # 计算立方体的中心和尺寸
        center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]
        size = [(max_corner[i] - min_corner[i]) for i in range(3)]

        # 添加立方体
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        cube = bpy.context.object
        cube.scale = [s / 2 for s in size]  # 调整立方体的尺寸

        # 创建材质
        mat = bpy.data.materials.new(name="TransparentMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = cube_color  # 设置为半透明
        bsdf.inputs['Alpha'].default_value = cube_color[-1]  # 设置为半透明
        mat.blend_method = 'BLEND'  # 设置混合模式为混合

        # 将材质赋予对象
        cube.data.materials.append(mat)

        # 添加 Wireframe 修饰符以创建边框
        bpy.ops.object.modifier_add(type='WIREFRAME')
        cube.modifiers["Wireframe"].thickness = 0.05
        cube.modifiers["Wireframe"].use_replace = False  # 保留原始几何体

        # 创建一个新的材质用于棱边
        edge_mat = bpy.data.materials.new(name="EdgeMaterial")
        edge_mat.use_nodes = True
        edge_bsdf = edge_mat.node_tree.nodes.get("Principled BSDF")
        edge_bsdf.inputs['Base Color'].default_value = edge_color  # 设置棱边颜色
        edge_bsdf.inputs['Alpha'].default_value = 1.0  # 不透明
        edge_mat.blend_method = 'BLEND'  # 设置混合模式为混合

        # 将棱边材质赋予对象
        if len(cube.data.materials) < 2:
            cube.data.materials.append(edge_mat)
        else:
            cube.data.materials[1] = edge_mat

        # 设置对象的显示模式为线框
        cube.display_type = 'WIRE'
        
        # 设置圆锥的实例 ID
        inst_id = len(self.objects) + 1000
        cube["inst_id"] = inst_id
        self.objects[f'Cube_{id}'] = cube

    def add_axes(self):
        """
        在场景中添加一个坐标轴。
        """
        # X 轴
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1.5, location=(1, 0, 0))
        x_axis = bpy.context.object
        x_axis.rotation_euler = (0, 0, 90)
        x_axis.name = 'X_Axis'

        # Y 轴
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1.5, location=(0, 1, 0))
        y_axis = bpy.context.object
        y_axis.rotation_euler = (90, 0, 0)
        y_axis.name = 'Y_Axis'

        # Z 轴
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1.5, location=(0, 0, 1))
        z_axis = bpy.context.object
        z_axis.name = 'Z_Axis'

        # 添加材质
        for axis, color in zip([x_axis, y_axis, z_axis], [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]):
            mat = bpy.data.materials.new(name=f'{axis.name}_Material')
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Alpha'].default_value = 0.5 
            axis.data.materials.append(mat)

    def rotate_objects_z(self):
        """
        将场景中的所有对象绕 Z 轴旋转 90 度。
        """
        rotation = math.radians(-90)
        for obj in bpy.context.scene.objects:
            if obj.type in {'MESH', 'EMPTY'}:  # 可选择限制类型
                obj.rotation_euler.rotate_axis("Y", rotation)

    def add_light(self):
        """
        向场景中添加一个光源。
        """
        bpy.ops.object.light_add(type='SUN', radius=1, location=(5, 5, 5))
        light = bpy.context.object
        light.data.energy = 10  # 调整光源强度

if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = 'ShapeNetScene'
    print(cfg.env.scene)

    # 创建环境
    target_path = "/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133"
    obj_file_path = target_path+'/models/model_normalized.obj'
    json_path = obj_file_path[:-4]+'.json'
    with open(json_path, 'r') as file:
        data = json.load(file)
        centroid = data.get("centroid", None)
    # obj_ = bpy.ops.import_scene.obj(filepath=obj_file_path)
    # mesh = trimesh.load(obj_file_path)
    cfg.env.target_path = obj_file_path # type: ignore
    scene = eval(cfg.env.scene)(cfg=cfg.env)
    
    # 渲染图像
    camera_position = mathutils.Vector((1, 1, 1))
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y')) # wxyz
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position))) # xyzwxyz
    img = scene.render_pose(fixed_pose) # RGBA
    save_img(img[:,:,:3], 'data/images/scene_test.png')

    
    # gen data test
    poses = scene.gen_data_fn['full']()
    # print(f'init mode poses: {poses}')
    for i,pose in enumerate(poses):
        # scene = eval(cfg.env.scene)(cfg=cfg.env)
        img = scene.render_pose(pose)
        save_img(img[:,:,:3], f'data/images/scene_gendata_init_{i}.png')
    # images = scene.render_poses(poses)
    # for i,image in enumerate(images):
    #     save_img(image[:,:,:3], f'images/scene_gendata_init_{i}.jpg')
    
    # add cone test
    camera_position2 = mathutils.Vector((0, 0, 1))
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position2
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y')) # wxyz
    scene = eval(cfg.env.scene)(cfg=cfg.env)
    scene.add_cone_at_pose2(camera_position2, rot_quat)

    img = scene.render_pose(fixed_pose) # RGBA
    save_img(img[:,:,:3], 'data/images/add_cone_test.png')


    mesh = scene.get_mesh()
    print(f'mesh:{mesh}')
    aabb = scene.get_aabb()
    print(f'aabb:{aabb}')

    print('test success')