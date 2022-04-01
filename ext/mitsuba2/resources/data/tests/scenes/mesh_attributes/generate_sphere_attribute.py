import numpy as np

import mitsuba
mitsuba.set_variant("scalar_rgb")

from mitsuba.core import xml, Thread

Thread.thread().file_resolver().append(os.path.dirname(__file__ ) + '../../../common')

m = xml.load_string("""
    <shape type="ply" version="2.0.0">
        <string name="filename" value="meshes/sphere.ply"/>
    </shape>
""")

attribute = m.add_attribute("face_weight", 1, np.random.rand(m.face_count()))
attribute = m.add_attribute("face_color", 3, np.random.rand(3 * m.face_count()))
attribute = m.add_attribute("vertex_color", 3, np.random.rand(3 * m.vertex_count()))
m.write_ply("meshes/sphere_attribute.ply")
