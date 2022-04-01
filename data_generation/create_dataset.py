import os
import sys
import configargparse
import random
import time
import numpy as np
import multiprocessing
from multiprocessing import Pool

sys.path += ["..\\ext\\mitsuba2\\dist\\python"]
os.environ["PATH"] += os.pathsep + "..\\ext\\mitsuba2\\dist"

import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('gpu_rgb')

import enoki

from neural_rendering.utils import create_dir
from data_generation.variable_renderer import write_image
from data_generation.variable_renderer import write_buffers
from data_generation.variable_renderer import write_variable
from data_generation.variable_renderer import VariableRenderer


def initialize_process(scene_path, dataset_path_, tonemap):
    global renderer

    renderer = VariableRenderer(tonemap_type=tonemap)
    renderer.load_scene(scene_path)

    global dataset_path

    dataset_path = dataset_path_


def run_process(id):
    custom_values = dict()

    # Get parameters from Markov Chain
    for j in range(len(renderer.variables_ids)):
        var_id = renderer.variables_ids[j]
        parameters = []
        for k in range(renderer.variables[j].num_parameters()):
            parameters.append(random.uniform(0, 1))

        write_variable(parameters, dataset_path, var_id)
        custom_values[var_id] = parameters

    buffers, gt, custom_values = renderer.get_custom_render(custom_values, need_buffers=True, need_image=True)

    return buffers, gt


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    conf = configargparse.ArgumentParser()

    conf.add('--dataset_dir', default='../datasets', help='Path to save the generated dataset')
    conf.add('--num_samples', type=int, required=True, help='Number of generated samples')
    conf.add('--scene_path', required=True, help='Path to the scene used for the dataset generation')
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--tonemap', default='log1p', choices=['log1p'])

    conf = conf.parse_args()

    random.seed(conf.seed)

    # Create renderer
    renderer = VariableRenderer(tonemap_type=conf.tonemap)

    # Load scene
    renderer.load_scene(conf.scene_path)

    resolution = renderer.sensors[0].film().size()

    dataset_path = os.path.join(conf.dataset_dir, 'dataset_' + time.strftime("%Y%m%d-%H%M%S"))
    create_dir(dataset_path)

    pool = Pool(initializer=initialize_process, initargs=[conf.scene_path, dataset_path, conf.tonemap], processes=1)

    for i in range(conf.num_samples):

        before = pool._pool[:]

        # Render inputs in a process to handle the possible crashes due to memory issues
        results = pool.map_async(run_process, range(1))

        # Wait for parallel results
        while not results.ready():
            # If one of the processes crashed repeat sample generation
            if any(proc.exitcode for proc in before):
                # Release unused memory
                enoki.cuda_malloc_trim()
                # Restart pool
                pool.terminate()
                pool.join()
                pool = Pool(initializer=initialize_process, initargs=[conf.scene_path, dataset_path, resolution[0], conf.patch_size, conf.tonemap, conf.setup_granskog], processes=1)
                break
        else:

            results = results.get()

            buffers = results[0][0]
            gt = results[0][1]

            np.savez(dataset_path + '/' + str(i) + 'sample.npz', *buffers, gt)

            print('Dataset generation -- %d/%d' % (i, conf.num_samples))

            i += 1







