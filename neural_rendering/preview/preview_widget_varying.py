import sys
import os

sys.path += ["..\\ext\\mitsuba2\\dist\\python"]
os.environ["PATH"] += os.pathsep + "..\\ext\\mitsuba2\\dist"

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
import time
from imgui.integrations.glfw import GlfwRenderer
from neural_rendering.generators.pixel_generator import PixelGenerator
from neural_rendering.generators.positional_pixel_generator import PositionalPixelGenerator
from neural_rendering.utils import *
from neural_rendering.losses import AllMetrics
from data_generation.tonemap import *

diff_exposure = 1
exposure = [0.5]
preview_resolution = 600
resolution = 300


def preview():
    conf = configargparse.ArgumentParser()

    conf.add('--model_path', required=True, help='Path to model which will be used for tha path generation')
    conf.add('--scene_path', required=True, help='Path to the scene to be preview')
    conf.add('--scene_buffers_path', required=True, help='Path to the buffers version of scene to preview')

    # Generators (default: Pixel Generator)
    conf.add('--arch', default='pixel', choices=['pixel', 'ppixel'])
    conf.add('--device', type=str, default='cuda', help='Cuda device to use')
    conf.add('--tonemap', default='log1p', choices=['log1p'])
    conf.add('--metric', default='dssim', choices=['l1', 'l2', 'lpips', 'dssim', 'mape', 'smape', 'mrse'])
    conf.add('--hidden_features', type=int, default=700, help='Number of hidden features for the generator')
    conf.add('--hidden_layers', type=int, default=8, help='Number of hidden layers for the generator')

    # Set random seeds
    random.seed(0)
    torch.manual_seed(0)

    conf = conf.parse_args()

    renderer = VariableRenderer(tonemap_type=conf.tonemap)
    renderer_buffers = VariableRenderer(tonemap_type=conf.tonemap)

    # Set inverse tonemapping
    if conf.tonemap == 'log1p':
        inv_tonemap = inv_log1p

    # Losses for visualization
    criterion = AllMetrics()

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

    model_path = conf.model_path

    if conf.arch == 'pixel':
        print("Using Pixel generator")
        model = PixelGenerator(buffers_features=13, variables_features=renderer.scene.total_parameters(), hidden_features=conf.hidden_features, hidden_layers=conf.hidden_layers)
    elif conf.arch == 'ppixel':
        print('Using Positional Pixel generator')
        model = PositionalPixelGenerator(buffers_features=13, variables_features=renderer.scene.total_parameters(), hidden_features=conf.hidden_features, hidden_layers=conf.hidden_layers)

    # Load model
    model.half()
    model.cuda()

    model.load_state_dict(torch.load(model_path, map_location=conf.device))

    # Initialize window
    window = impl_glfw_init()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize images
    prediction_img = np.zeros((preview_resolution, preview_resolution, 3)).astype(np.float16)
    gt_img = np.zeros((resolution, resolution, 3))
    closest_img = np.zeros((resolution, resolution, 3))
    diff_img = np.zeros((resolution, resolution, 3))

    # Bind prediction texture
    prediction_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, prediction_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16F, preview_resolution, preview_resolution, 0, gl.GL_RGB, gl.GL_FLOAT, prediction_img)

    # Bind ground truth texture
    gt_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, gt_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, gt_img)

    # Bind closest data point texture
    closest_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, closest_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, closest_img)

    # Bind diff texture
    diff_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, diff_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, diff_img)

    # Enable key event callback
    glfw.set_key_callback(window, key_event)

    # Images are in linear space transform them to sRGB
    gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
    gl.glEnable(gl.GL_DITHER)

    loss = 0

    frames = 0
    fps = 0

    start_frame = time.time()

    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.set_next_window_position(600, 0)
        imgui.set_next_window_size(600, 300)
        imgui.begin("Configurable parameters", flags=imgui.WINDOW_NO_MOVE)

        imgui.text('FPS: ' + str(fps))

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

        imgui.spacing()

        imgui.end()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(600, 920)
        imgui.begin("Preview", flags=imgui.WINDOW_NO_MOVE)

        # Draw Neural Prediction
        with torch.no_grad():
            buffers, _, custom_values = renderer_buffers.get_custom_render_tensor(custom_values, need_buffers=True, need_image=False)

            inputs = stack_inputs_tensor(buffers, renderer_buffers.variables, [*custom_values.values()])
            inputs = inputs.half()
            inputs = inputs.unsqueeze(0)

            prediction = model(inputs)
            prediction = prediction.detach()[0, :, :, :]
            prediction = inv_tonemap(prediction.float().cpu().numpy())
            prediction = cv2.resize(prediction, (preview_resolution, preview_resolution), cv2.INTER_NEAREST)

            prediction_img = prediction * exposure[0]

            gl.glBindTexture(gl.GL_TEXTURE_2D, prediction_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, preview_resolution, preview_resolution, gl.GL_RGB, gl.GL_FLOAT, prediction_img)

        imgui.image(prediction_id, preview_resolution, preview_resolution)

        # Draw Ground Truth and Diff
        imgui.begin_group()

        if imgui.button("Generate GT"):
            _, gt, custom_values = renderer.get_custom_render(custom_values, need_buffers=False, need_image=True)
            gt = inv_tonemap(gt)
            gt = cv2.resize(gt, (resolution, resolution), cv2.INTER_NEAREST)

            gt_img = gt * exposure[0]

            gl.glBindTexture(gl.GL_TEXTURE_2D, gt_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution, resolution, gl.GL_RGB, gl.GL_FLOAT, gt_img)

            loss = criterion(torch.from_numpy(prediction).unsqueeze(0).to(conf.device), torch.from_numpy(cv2.resize(gt, (preview_resolution, preview_resolution), cv2.INTER_NEAREST)).unsqueeze(0).to(conf.device)).metrics[conf.metric].mean().item()

            # If metric can be visualized
            if criterion.metrics[conf.metric].dim() > 1:
                diff_img = np.mean(criterion.metrics[conf.metric][0, :, :, :].detach().cpu().numpy(), axis=2, keepdims=False)
                diff_img - diff_img.min() / (diff_img.max() - diff_img.min())

                # Apply Google Turbo colormap
                diff_img = apply_colormap(turbo_colormap_data, diff_img * diff_exposure)
                diff_img = cv2.resize(diff_img, (resolution, resolution), cv2.INTER_NEAREST)

                gl.glBindTexture(gl.GL_TEXTURE_2D, diff_id)
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution, resolution, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, diff_img)

        imgui.image(gt_id, resolution, resolution)

        imgui.end_group()

        imgui.same_line()

        imgui.begin_group()

        imgui.text('loss : %.4f' % loss)

        imgui.image(diff_id, resolution, resolution)

        imgui.end_group()

        imgui.end()

        gl.glClearColor(0.00, 0.00, 0.00, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        frames += 1

        if (time.time() - start_frame) > 1.0:
            fps = frames
            frames = 0
            start_frame = time.time()

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1200, 920
    window_name = "Neural Rendering"

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
