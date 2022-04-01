import numpy as np
import cv2

from mitsuba.core import Transform4f, Bitmap, Vector3f, UInt32, AnimatedTransform, ScalarTransform4f
from enoki.scalar import Vector3f as sVector3f

import enoki as ek


# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim=3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))


def ravel_numpy(buf, dim=3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    return np.column_stack([ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2)])


# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim=3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)


def set_parameter(params, v, id):
    params[id] = v
    params.update()


def set_translation(old_params, new_params, v, id):
    trasfo = Transform4f.translate(v)
    positions_buf = old_params[id]
    positions_initial = ravel(positions_buf)
    new_positions = trasfo.transform_point(positions_initial)
    unravel(new_positions, new_params[id])
    new_params.set_dirty(id)
    new_params.update()


def set_sensor(sensor, origin, target, up=[0, 1, 0]):
    sensor.set_world_transform(AnimatedTransform(ScalarTransform4f.look_at(sVector3f(origin[0], origin[1], origin[2]),
                                                                           sVector3f(target[0], target[1], target[2]),
                                                                           sVector3f(up[0], up[1], up[2]))))


def get_values(params, id):
    value_buf = params[id]
    value_initial = ravel(value_buf)
    return value_initial


def get_value(params, id):
    value_buf = params[id]
    return value_buf


def apply_translation(params, v, id):
    trasfo = Transform4f.translate(v)
    positions_buf = params[id]
    positions_initial = ravel(positions_buf)
    new_positions = trasfo.transform_point(positions_initial)
    unravel(new_positions, params[id])


def apply_translation_from(params, v, positions_initial, id):
    trasfo = Transform4f.translate(v)
    new_positions = trasfo.transform_point(positions_initial)
    unravel(new_positions, params[id])
    params.set_dirty(id)

def write_image(gt, iteration, directory):
    cv2.imwrite(directory + '/' + str(iteration) + 'image_output.exr', gt)


def write_buffers(buffers, iteration, directory):
    cv2.imwrite(directory + '/' + str(iteration) + 'albedo_output.exr', buffers[:, :, 0:3])
    cv2.imwrite(directory + '/' + str(iteration) + 'normals_output.exr', buffers[:, :, 3:6])
    cv2.imwrite(directory + '/' + str(iteration) + 'positions_output.exr', buffers[:, :, 6:9])
    cv2.imwrite(directory + '/' + str(iteration) + 'wi_output.exr', buffers[:, :, 9:12])
    cv2.imwrite(directory + '/' + str(iteration) + 'emission_output.exr', buffers[:, :, 12:15])
    cv2.imwrite(directory + '/' + str(iteration) + 'alpha_output.exr', buffers[:, :, 15:16])


def write_variable(variable, directory, name):
    f = open(directory + '/' + name + '.txt', 'a')
    for i in range(len(variable)):
        f.write(str(variable[i]) + " ")
    f.write("\n")
    f.close()
