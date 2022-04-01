import warnings

from data_generation.utils import *
import os
import torch

from mitsuba.core import Bitmap, Struct
from mitsuba.core import Thread, LogLevel
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.core import ScalarTransform4f, ScalarVector3f, AnimatedTransform
from enoki.cuda import Vector3f as cVector3f
from mitsuba.python.autodiff import render_torch

from data_generation.tonemap import *


class VariableRenderer:

    def __init__(self, tonemap_type="log1p", setup_granskog=False):

        self.tonemap_type = tonemap_type
        self.setup_granskog = setup_granskog

        self.scene = None
        self.integrator = None
        self.params = None
        self.variables_ids = None
        self.variables = None
        self.sensors = None
        self.emitters = None
        self.shapes = None
        self.shapegroups = None
        self.bsdfs = None
        self.initial_values = None

    def load_scene(self, scene_filename):
        # Log only errors
        Thread.thread().logger().set_log_level(LogLevel.Error)

        # Add the scene directory to the FileResolver's search path
        Thread.thread().file_resolver().append(os.path.dirname(scene_filename))

        # Load the actual scene
        scene = load_file(scene_filename)

        # Get the scene parameters
        params = traverse(scene)

        # Scene randomizable objects
        emitters = scene.emitters()
        shapes = scene.shapes()
        bsdfs = scene.bsdfs()
        sensors = scene.sensors()
        shapegroups = scene.shapegroups()

        # Variables parameters
        variables = []
        variables_ids = []
        initial_values = []

        # Retrieve chosen variable objects
        for i in range(len(shapes)):
            if 'var' in shapes[i].id():
                variables_ids.append(shapes[i].id())
                variables.append(shapes[i])

                try:
                    initial_values.append(get_values(params, shapes[i].id() + '.vertex_positions_buf'))
                except:
                    pass

        for i in range(len(shapegroups)):
            if 'var' in shapegroups[i].id():
                variables_ids.append(shapegroups[i].id())
                variables.append(shapegroups[i])
                # Initial values not necessary for shapegroups

        for i in range(len(emitters)):
            if 'var' in emitters[i].id():
                variables_ids.append(emitters[i].id())
                variables.append(emitters[i])
                # Initial values not necessary for emitters

        for i in range(len(bsdfs)):
            if 'var' in bsdfs[i].id():
                variables_ids.append(bsdfs[i].id())
                variables.append(bsdfs[i])
                # Initial values not necessary for bsdfs

        for i in range(len(sensors)):
            if 'var' in sensors[i].id():
                variables_ids.append(sensors[i].id())
                variables.append(sensors[i])
                # Initial values not necessary for sensors

        variable_params = []

        # Save the params that are variable and keep only those in the parameter map
        for i in range(len(variables)):
            if variables[i] in emitters:
                param_id = variables[i].id() + '.radiance.value'
                variable_params.append(param_id)

            elif variables[i] in shapes:
                # Handle shape group instances
                if variables[i].is_instance():
                    param_id = variables[i].id() + '.to_world'
                    variable_params.append(param_id)
                # Handle single shapes
                else:
                    param_id = variables[i].id() + '.vertex_positions_buf'
                    variable_params.append(param_id)

            elif variables[i] in bsdfs:
                if variables[i].is_glossy():
                    param_id = variables[i].id() + '.specular_reflectance.value'
                    variable_params.append(param_id)

                    param_id = variables[i].id() + '.diffuse_reflectance.value'
                    variable_params.append(param_id)

                    param_id = variables[i].id() + '.alpha.value'
                    variable_params.append(param_id)

                    param_id = variables[i].id() + '.alpha'
                    variable_params.append(param_id)

        params.keep(variable_params)

        self.scene = scene
        self.integrator = scene.integrator()
        self.params = params
        self.variables_ids = variables_ids
        self.variables = variables
        self.sensors = sensors
        self.emitters = emitters
        self.shapes = shapes
        self.shapegroups = shapegroups
        self.bsdfs = bsdfs
        self.initial_values = initial_values

    def setup_scene(self, custom_values):
        for i in range(len(self.variables)):

            # Emitters
            if self.variables[i] in self.emitters:
                assert self.variables[i].num_parameters() in [1, 3], "Emitters need 1 or 3 parameters, defined in the xml as num_parameters"

                if self.variables[i].is_environment():
                    # TODO: WIP implementation of rotating environment map, currently it is rotated in a fixed range
                    self.variables[i].set_world_transform(AnimatedTransform(ScalarTransform4f.rotate(ScalarVector3f(0.0, 1.0, 0.0), 100 + custom_values[self.variables[i].id()][0] * 40)))
                else:
                    param_id = self.variables[i].id() + '.radiance.value'

                    # Change the intensity of emission
                    if self.variables[i].num_parameters() == 1:
                        self.params[param_id] = self.variables[i].min_bounds() + cVector3f(custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][0]) * self.variables[i].range_bounds()

                    # Change X, Y, Z of emission individually
                    if self.variables[i].num_parameters() == 3:
                        self.params[param_id] = self.variables[i].min_bounds() + cVector3f(custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1], custom_values[self.variables[i].id()][2]) * self.variables[i].range_bounds()

            # Sensors
            elif self.variables[i] in self.sensors:
                assert self.variables[i].num_parameters() in [5], "Sensors need 5 parameters, defined in the xml as num_parameters"

                bbox_range = self.scene.bbox().extents()

                pos = self.variables[i].min_bounds() + cVector3f(custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1],
                                                                 custom_values[self.variables[i].id()][2]) * self.variables[i].range_bounds()

                origin = np.array([pos[0], pos[1], pos[2]])

                if self.variables[i].num_parameters() == 5:
                    # Target ranges based on bbox
                    target = np.array([self.scene.bbox().min[0] + bbox_range.x / 3 + (custom_values[self.variables[i].id()][3] * bbox_range.x / 3),
                                       self.scene.bbox().min[1] + (bbox_range.y / 3),
                                       self.scene.bbox().min[2] + bbox_range.z / 3 + (custom_values[self.variables[i].id()][4] * bbox_range.z / 3)])

                set_sensor(self.variables[i], origin, target)

            # Shapes
            elif self.variables[i] in self.shapes:
                assert self.variables[i].num_parameters() in [1, 2, 3], "Shapes need 1, 2 or 3 parameters, defined in the xml as num_parameters"

                param_index = 0

                custom_vector = [0, 0, 0]

                # If x has a range use it as a parameter
                if self.variables[i].range_bounds().x[0] > 0:
                    custom_vector[0] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                # If y has a range use it as a parameter
                if self.variables[i].range_bounds().y[0] > 0:
                    custom_vector[1] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                # If z has a range use it as a parameter
                if self.variables[i].range_bounds().z[0] > 0:
                    custom_vector[2] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                translate = self.variables[i].min_bounds() + cVector3f(custom_vector[0], custom_vector[1], custom_vector[2]) * self.variables[i].range_bounds()

                # Handle shape group instances
                if self.variables[i].is_instance():
                    param_id = self.variables[i].id() + '.to_world'

                    # The parameter left is for rotation
                    if self.variables[i].num_parameters() > param_index:
                        rotation_axis = ScalarVector3f(self.variables[i].rotation_axis()[0][0], self.variables[i].rotation_axis()[1][0], self.variables[i].rotation_axis()[2][0])
                        angle = (self.variables[i].min_angle() + custom_values[self.variables[i].id()][param_index] * self.variables[i].range_angle())[0]

                        self.variables[i].set_to_world(ScalarTransform4f.translate(ScalarVector3f(translate.x[0], translate.y[0], translate.z[0])) * ScalarTransform4f.rotate(rotation_axis, angle))
                    # Only translate
                    else:
                        self.variables[i].set_to_world(ScalarTransform4f.translate(ScalarVector3f(translate.x[0], translate.y[0], translate.z[0])))

                    self.params.set_dirty(param_id)
                # Handle single shapes
                else:
                    param_id = self.variables[i].id() + '.vertex_positions_buf'

                    apply_translation_from(self.params, self.initial_values[i], translate, param_id)

            # Shapegroups
            elif self.variables[i] in self.shapegroups:
                assert self.variables[i].num_parameters() in [1], "Shape groups need 1 defined in the xml as num_parameters"

                self.variables[i].set_alternative(int(min(0.99, custom_values[self.variables[i].id()][0]) * (self.variables[i].num_alternatives() + 1)))
                custom_values[self.variables[i].id()][0] = int(min(0.99, custom_values[self.variables[i].id()][0]) * (self.variables[i].num_alternatives() + 1)) / (self.variables[i].num_alternatives() + 1)

            # BSDFs
            elif self.variables[i] in self.bsdfs:
                min_bounds = self.variables[i].min_bounds()
                range_bounds = self.variables[i].range_bounds()

                # Glossy BSDFs
                if self.variables[i].is_glossy():
                    assert self.variables[i].num_parameters() in [1, 3, 4], "Glossy BSDFs can have 1, 3 or 4 parameters"

                    # Set variable reflectance
                    if self.variables[i].num_parameters() in [3, 4]:
                        specular_reflectance = cVector3f(min_bounds[0], min_bounds[1], min_bounds[2]) + cVector3f(custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1], custom_values[self.variables[i].id()][2]) * cVector3f(range_bounds[0], range_bounds[1], range_bounds[2])

                        param_id = self.variables[i].id() + '.specular_reflectance.value'

                        self.params[param_id] = specular_reflectance

                    # Set variable roughness
                    if self.variables[i].num_parameters() in [1, 4]:
                        alpha = [min_bounds[3] + custom_values[self.variables[i].id()][self.variables[i].num_parameters() - 1] * range_bounds[3]][0]

                        # TODO: trying different keys because rough conductor and rough plastic use different keys for alpha

                        # Rough conductor
                        try:
                            param_id = self.variables[i].id() + '.alpha.value'

                            self.params[param_id] = alpha
                        except:
                            pass

                        # Rough plastic
                        try:
                            param_id = self.variables[i].id() + '.alpha'

                            self.params[param_id] = alpha[0]
                        except:
                            pass

                # Diffuse BSDFs
                else:
                    assert self.variables[i].num_parameters() in [1, 3, 4], "Diffuse BSDFs can have 1, 3 or 4 parameters"

                    reflectance = cVector3f(min_bounds[0], min_bounds[1], min_bounds[2]) + cVector3f(custom_values[self.variables[i].id()][0], custom_values[self.variables[i].id()][1], custom_values[self.variables[i].id()][2]) * cVector3f(range_bounds[0], range_bounds[1], range_bounds[2])

                    # Set variable reflectance
                    if self.variables[i].num_parameters() in [3, 4]:
                        self.variables[i].set_modifier(reflectance)

                    # Set texture index
                    if self.variables[i].num_parameters() in [1, 4]:
                        self.variables[i].set_alternative(int(custom_values[self.variables[i].id()][3] * self.variables[i].num_alternatives()))

        self.params.update()

        return custom_values

    def get_custom_render(self, custom_values, need_image=True, need_buffers=True):

        # Set up the scene for the given custom  values and check intersection
        custom_values = self.setup_scene(custom_values)

        # Call the scene's integrator to render the loaded scene
        self.integrator.render(self.scene, self.sensors[0])

        # After rendering, get the aovs stored in the film
        components = self.scene.sensors()[0].film().bitmap(raw=False).split()

        buffers = []
        gt = []

        # Choose tonemapping for gt and emission
        if self.tonemap_type == 'log1p':
            tonemap = log1p
        else:
            warnings.warn("Chosen tonemapping not supported")

        for i in range(len(components)):
            if 'image' in components[i][0]:
                if need_image:
                    gt = components[i][1].convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
                    gt = gt.data_np()
                    gt = tonemap(gt)
            elif need_buffers:
                if 'position' in components[i][0]:
                    buffer = components[i][1].convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
                    buffer = buffer.data_np()
                    # Normalize from bounding box
                    buffer = (buffer - self.scene.bbox().min) / (self.scene.bbox().max - self.scene.bbox().min)
                    buffers.append(buffer)
                elif 'normal' in components[i][0]:
                    buffer = components[i][1].convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
                    buffer = buffer.data_np()
                    # Normalize between 0 and 1
                    buffer = (buffer / 2.0) + 0.5
                    buffers.append(buffer)
                elif 'albedo' in components[i][0]:
                    buffer = components[i][1].convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
                    buffer = buffer.data_np()
                    buffers.append(buffer)
                elif 'wi' in components[i][0]:
                    buffer = components[i][1].convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
                    buffer = buffer.data_np()
                    buffers.append(buffer)
                elif 'emission' in components[i][0]:
                    buffer = components[i][1].convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
                    buffer = buffer.data_np()
                    # Same tonemap as gt
                    buffer = tonemap(buffer)
                    buffers.append(buffer)
                elif 'alpha' in components[i][0]:
                    buffer = components[i][1].convert(Bitmap.PixelFormat.Y, Struct.Type.Float32, srgb_gamma=False)
                    buffer = buffer.data_np()
                    buffers.append(buffer)

        return buffers, gt, custom_values

    def get_custom_render_tensor(self, custom_values, need_image=True, need_buffers=True):

        # Setup the scene for the given custom  values and check intersection
        custom_values = self.setup_scene(custom_values)

        # Call the scene's integrator to render the loaded scene
        result = render_torch(self.scene)

        buffers = torch.tensor(data=[], device='cuda')

        gt = []

        # Choose tonemapping for gt and emission
        if self.tonemap_type == 'log1p':
            tonemap = log1p_tensor
        else:
            warnings.warn("Chosen tonemapping not supported")

        if need_image:
            gt = result[:, :, 0:3]
            gt = tonemap(gt)

        if need_buffers:

            emission = result[:, :, 3:6]
            emission = tonemap(emission)

            normals = result[:, :, 6:9]
            # Normalize between 0 and 1
            normals = (normals / 2.0) + 0.5

            positions = result[:, :, 9:12]
            # Normalize from bounding box
            bbox_range = self.scene.bbox().max - self.scene.bbox().min
            positions = (positions - torch.tensor([self.scene.bbox().min.x, self.scene.bbox().min.y, self.scene.bbox().min.z], device='cuda')) / torch.tensor([bbox_range.x, bbox_range.y, bbox_range.z], device='cuda')

            wi = result[:, :, 12:15]

            albedo = result[:, :, 15:18]

            alpha = result[:, :, 18:19]

            buffers = [emission, normals, positions, wi, albedo, alpha]

        return buffers, gt, custom_values
