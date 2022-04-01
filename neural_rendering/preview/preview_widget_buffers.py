import sys
import os

sys.path += ["..\\..\\ext\\mitsuba2\\dist\\python"]
os.environ["PATH"] += os.pathsep + "..\\..\\ext\\mitsuba2\\dist"

import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('gpu_rgb')

import torch
import random
from data_generation.variable_renderer import *
import configargparse
import cv2
import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
from neural_rendering.utils import *
from data_generation.tonemap import *

exposure = [1]
resolution = 600


def preview():
    conf = configargparse.ArgumentParser()

    conf.add('--scene_path', required=True, help='Path to the scene to be preview')
    conf.add('--scene_buffers_path', required=True, help='Path to the buffers version of scene to preview')
    conf.add('--tonemap', default='log1p', choices=['log1p'])

    # Set random seeds
    random.seed(0)
    torch.manual_seed(0)

    conf = conf.parse_args()

    renderer = VariableRenderer(tonemap_type=conf.tonemap)
    renderer_buffers = VariableRenderer(tonemap_type=conf.tonemap)

    # Set inverse tonemapping
    if conf.tonemap == 'log1p':
        inv_tonemap = inv_log1p

    # Load two versions of the scene to avoid rendering the image when only needing the buffers and vice versa
    renderer.load_scene(conf.scene_path)
    renderer_buffers.load_scene(conf.scene_buffers_path)

    custom_values = dict()

    # Dynamically create the dataset
    for i in range(len(renderer.variables_ids)):
        var_id = renderer.variables_ids[i]
        parameters = []
        for k in range(renderer.variables[i].num_parameters()):
            parameters.append(random.uniform(0, 1))

        custom_values[var_id] = parameters

    # Initialize window
    window = impl_glfw_init()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize images
    buffer_img = np.zeros((resolution, resolution, 3))
    gt_img = np.zeros((resolution, resolution, 3))

    # Bind buffer texture
    buffer_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, buffer_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, buffer_img)

    # Bind ground truth texture
    gt_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, gt_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, gt_img)

    # Enable key event callback
    glfw.set_key_callback(window, key_event)

    buffer_index = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(1200, 920)
        imgui.begin("Configurable parameters", flags=imgui.WINDOW_NO_MOVE)

        imgui.push_item_width(500)

        # Sensor sliders
        for i in range(len(renderer.variables_ids)):
            if renderer.variables[i] in renderer.sensors:
                var_id = renderer.variables_ids[i]
                if renderer.variables[i].num_parameters() == 1:
                    changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                    values = [values]
                elif renderer.variables[i].num_parameters() == 2:
                    changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 3:
                    changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 4:
                    changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 5:
                    changed, values = imgui.slider_float3(var_id + '_origin', *custom_values[var_id][0:3], 0, 1)
                    changed2, values2 = imgui.slider_float2(var_id + '_target', *custom_values[var_id][3:5], 0, 1)
                    values = values + values2
                elif renderer.variables[i].num_parameters() == 6:
                    changed, values = imgui.slider_float3(var_id + '_origin', *custom_values[var_id][0:3], 0, 1)
                    changed2, values2 = imgui.slider_float3(var_id + '_target', *custom_values[var_id][3:6], 0, 1)
                    values = values + values2

                custom_values[var_id] = list(values)

        # Emitters sliders
        for i in range(len(renderer.variables_ids)):
            if renderer.variables[i] in renderer.emitters:
                var_id = renderer.variables_ids[i]
                if renderer.variables[i].num_parameters() == 1:
                    changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                    values = [values]
                elif renderer.variables[i].num_parameters() == 2:
                    changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 3:
                    changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 4:
                    changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)

                custom_values[var_id] = list(values)

        # Shapes sliders
        for i in range(len(renderer.variables_ids)):
            if renderer.variables[i] in renderer.shapes:
                var_id = renderer.variables_ids[i]
                if renderer.variables[i].num_parameters() == 1:
                    changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                    values = [values]
                elif renderer.variables[i].num_parameters() == 2:
                    changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 3:
                    changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 4:
                    changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)

                custom_values[var_id] = list(values)

        # Shape groups sliders
        for i in range(len(renderer.variables_ids)):
            if renderer.variables[i] in renderer.shapegroups:
                var_id = renderer.variables_ids[i]
                if renderer.variables[i].num_parameters() == 1:
                    changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                    values = [values]
                elif renderer.variables[i].num_parameters() == 2:
                    changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)

                custom_values[var_id] = list(values)

        # BSDFs sliders
        for i in range(len(renderer.variables_ids)):
            if renderer.variables[i] in renderer.bsdfs:
                var_id = renderer.variables_ids[i]
                if renderer.variables[i].num_parameters() == 1:
                    changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                    values = [values]
                elif renderer.variables[i].num_parameters() == 2:
                    changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 3:
                    changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
                elif renderer.variables[i].num_parameters() == 4:
                    changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)

                custom_values[var_id] = list(values)

        if imgui.button("Previous Buffer"):
            buffer_index = buffer_index - 1
            # Loop buffers index
            if buffer_index < 0:
                buffer_index += 6

        imgui.same_line()

        if imgui.button("Next Buffer"):
            buffer_index = (buffer_index + 1) % 6

        imgui.same_line()

        # Draw Ground Truth
        if imgui.button("Generate GT"):
            _, gt, custom_values = renderer.get_custom_render(custom_values, need_buffers=False, need_image=True)
            gt_img = gt * exposure[0]
            gt_img = linear_to_srgb(gt_img)
            gt_img = np.clip(gt_img * 255, 0, 255)

            gt_img = cv2.resize(gt_img, (resolution, resolution), cv2.INTER_CUBIC)

            gl.glBindTexture(gl.GL_TEXTURE_2D, gt_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution, resolution, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, gt_img)

        # Draw Buffers
        with torch.no_grad():
            buffers, _, custom_values = renderer_buffers.get_custom_render(custom_values, need_buffers=True, need_image=False)

            if buffer_index == 5:
                buffer = buffers[5]

                buffer_img = np.clip(buffer * exposure[0] * 255, 0, 255)

                buffer_img = cv2.resize(buffer_img, (resolution, resolution), cv2.INTER_CUBIC)
                buffer_img = cv2.cvtColor(buffer_img, cv2.COLOR_GRAY2RGB)
            else:
                buffer = buffers[buffer_index]

                buffer_img = np.clip(buffer * exposure[0] * 255, 0, 255)

                buffer_img = cv2.resize(buffer_img, (resolution, resolution), cv2.INTER_CUBIC)

            gl.glBindTexture(gl.GL_TEXTURE_2D, buffer_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution, resolution, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, buffer_img)

        imgui.image(buffer_id, resolution, resolution)

        imgui.same_line()

        imgui.image(gt_id, resolution, resolution)

        imgui.end()

        gl.glClearColor(0.2, 0.2, 0.2, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1200, 920
    window_name = "Buffers Preview"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def key_event(window, key, scancode, action, mods):
    if (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_P:
        exposure[0] = exposure[0] + 0.1
    elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_O:
        exposure[0] = exposure[0] - 0.1


preview()
