# Variations

To define an element as variable it needs to have the `var_` tag in its id in all xml files. For each variable object you need to define the number of parameters through a `num_parameters` element and the range of each parameter through a `min_bounds` and `range_bounds` elements. These are used to sample the space of the variable. The currently supported variations are: 

## Variable Camera

### **Number of parameters: 5**

The variable camera has 5 parameters. Three of them control the x,y,z position of the camera and the two rest the x,z position of the target location. The range is defined only for the position of the camera. The range of the target location is computed from the bounding box of the scene. **Make sure that the scene bounding box is not huge due to unused objects**. For example a static camera that is defined like this:

```
<sensor type="perspective" id="sensor">
        <float name="fov" value="90" />
        <transform name="to_world">
            <matrix value="0.264209 0.071763 -0.961792 5.10518 -2.81996e-008 0.997228 0.074407 0.731065 0.964465 -0.019659 0.263476 -2.31789 0 0 0 1" />
        </transform>
        <sampler type="independent">
            <integer name="sample_count" value="400" />
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="128" />
            <integer name="height" value="128" />
            <rfilter type="tent" />
			<string name="component_format" value="float32"/>
        </film>
    </sensor>
```
Can be turned into a variable camera with a definition like this:
```
<sensor type="perspective" id="var_sensor">
        <float name="fov" value="90" />
        <transform name="to_world">
            <matrix value="0.264209 0.071763 -0.961792 5.10518 -2.81996e-008 0.997228 0.074407 0.731065 0.964465 -0.019659 0.263476 -2.31789 0 0 0 1" />
        </transform>
        <sampler type="independent">
            <integer name="sample_count" value="400" />
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="128" />
            <integer name="height" value="128" />
            <rfilter type="tent" />
			<string name="component_format" value="float32"/>
        </film>
		<vector name="min_bounds" value="0.4, 1.4, -3.1"/>
		<vector name="range_bounds" value="3.7, 0.3, 2.8"/>
		<integer name="num_parameters" value="5"/>
    </sensor>
```
## Shapes

### **Number of parameters: 1, 2 or 3**

A variable object can either be translated or rotated (not both). The `min_bounds` and `range_bounds` are used to define the range for translations. Rotations can be defined by assigning a rotation axis using the `rotation_axis` element and the `min_angle` and `range_angle` elements. Rotations always use 1 parameter. 
Example of variable object that can be translated:
```
<shape type="obj" id="var_small_box">
        <string name="filename" value="meshes/cbox_smallbox.obj" />
		<vector name="min_bounds" value="0, 1, 0"/>
		<vector name="range_bounds" value="0, 2, 0"/>
		<integer name="num_parameters" value="1"/>
        <ref id="box" />        
</shape>
```
Example of variable object that can be rotated:
```
<shape type="obj" id="var_small_box">
        <string name="filename" value="meshes/cbox_smallbox.obj" />
		<vector name="min_bounds" value="0, 0, 0"/>
		<vector name="range_bounds" value="0, 0, 0"/>
        <vector name="rotation_axis" value="0, 1, 0"/>
		<float name="range_angle" value="90"/>
		<float name="min_angle" value="-50"/>   
		<integer name="num_parameters" value="1"/>
        <ref id="box" />     
</shape>
```
## Shapegroup Instances

### **Number of parameters: 1, 2 or 3**

Shape group instances can be used to define variable group of objects. The variable parameters are the same as with shapes. To define a variable shapegroup instance you need to first define the shapegroup without any variability like so:
```
<shape type="shapegroup" id="blinds_group">
		<shape type="obj">
			<string name="filename" value="meshes/blind.obj" />
			<transform name="to_world">
				<matrix value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1" />
			</transform>
			<ref id="Blinds" />
		</shape>
		<shape type="obj">
			<string name="filename" value="meshes/blind2.obj" />
			<transform name="to_world">
				<matrix value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1" />
			</transform>
			<ref id="Blinds" />
		</shape>
	</shape>
```
And then create an instance of this shapegroup that is variable like so:
```
<shape type="instance" id="var_blinds">
    <ref id="blinds_group"/>
    <vector name="min_bounds" value="0, 0.3, 0"/>
    <vector name="range_bounds" value="0, 1.6, 0"/>
    <integer name="num_parameters" value="1"/>
</shape>
```

## Emitters

### **Number of parameters: 1 or 3**

Emitters can be defined variable to control their emission. The number of variables is either 1 or 3 depending whether we want to control the color of emission or the overall amount. An example of a variable emitter:
```
<emitter type="area" id="var_lamp_emitter">
        <rgb name="radiance" value="500,500,500" />
        <vector name="min_bounds" value="0, 0, 0"/>
        <vector name="range_bounds" value="500, 500, 500"/>	
        <integer name="num_parameters" value="1"/>
</emitter>
```

## Diffuse BSDFs

### **Number of parameters: 1, 3 or 4**

The variability for diffuse bsdfs includes changing the albedo and/or the texture. 3 parameters are used for variable albedo and 1 for changing texture index. An example of varying diffuse bsdf:

```
<bsdf type="diffuse" id="var_ceiling_mat">
    <rgb name="reflectance" value="1, 1, 1"/>
    <vector name="min_bounds" value="0, 0, 0"/>
    <vector name="range_bounds" value="1, 1, 1"/>
    <integer name="num_parameters" value="3"/>
</bsdf>
```
In order to have changing textures you first need to define the textures in the xml and then assign all of them to the same bsdf with the appropriate number of parameters. For example:
```
<bsdf type="diffuse" id="var_right_wall_mat">
    <ref id="wallpaper1_b" name="reflectance"/>
    <ref id="wallpaper2_b" name="reflectance"/>
    <ref id="wallpaper3" name="reflectance"/>
    <ref id="wallpaper4" name="reflectance"/>
    <vector name="min_bounds" value="0, 0, 0"/>
    <vector name="range_bounds" value="1, 1, 1"/>
    <integer name="num_parameters" value="4"/>
</bsdf>
```
## Glossy BSDFs

### **Number of parameters: 1, 3 or 4**

In glossy bsdfs you can define the reflectance and the roughness of the material as variable. 3 of the parameters are used for the color and 1 for the roughness. An example of a variable glossy bsdf:
```
<bsdf type="roughconductor" id="var_glossy_mat">
    <string name="distribution" value="ggx" />
    <rgb name="specular_reflectance" value="1, 1, 1"/>
    <integer name="num_parameters" value="4"/>
    <vector4f name="min_bounds" value="0, 0, 0, 0.5"/>
    <vector4f name="range_bounds" value="1, 1, 1, 0.5"/>
</bsdf>
```

## Combining Variations

Combining some variations can works but in many cases it might not work as intented. The main way to combine variations is by assigning variable materials to objects that can be moved. Or having a variable emmitter that can be moved or rotated. 

Other variations might be added in the future.
