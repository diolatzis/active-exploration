# Steps to compile and run

## Windows

1. Clone the repository with --recursive to get all the dependencies.
2. Compile Mitsuba 2, following the [documentation](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling.html) make sure to have all the prerequisites. 
3. Navigate to /ext/mitsuba2/ and run the command `cmake -G "Visual Studio 16 2019" -A x64`.
4. Create a PyCharm project from existing source at the main directory.

## Scene Definition

Any Mitsuba scene can be used to dynamically train the Pixel Generator. Given a normal Mitsuba scene,
you can define which objects are considered variable by adding the keyword `var` in the object id.
For example the object with id `small_box` becomes `var_small_box`. The types of objects currently
supported are shapes, point (light source) and diffuse bsdf. In the case of shapes and point light sources
what is randomized is the position and in the case of diffuse bsdf the albedo. In order to define the range
of values that these can take you can define the `min_bounds` and `max_bounds` parameters of the object 
(default: min_bounds = (0, 0, 0), max_bounds = (1, 1, 1)). For example:

```
<shape type="obj" id="var_small_box">
        <string name="filename" value="meshes/cbox_smallbox.obj" />
		<vector name="min_bounds" value="-1280, 80, 120"/>
		<vector name="max_bounds" value="3100, 0, 200"/>
        <ref id="box" />        
</shape>
```

The values of the object are then chosen as `v = min_bounds + random.uniform(0, 1)*max_bounds`.


## Scripts

The scripts are organized in 3 categories, neural_rendering, data_generation and example_1d. 
* example_1d contains the scripts with the 1d example configuration.
* data_generation contains the scripts that handle the rendering using Mitsuba 2.
* neural_rendering contains the scripts related to the training and the model.

#### Keywords

In the naming convention of the scripts certain keywords indicate certain functionalities.

* dynamic: The dataset is generated on the fly using Mitsuba 2
* markov: The exploration of the random numbers is done through MCMC
* uncertainty: The exporation of the random numbers is done through uncertainty (L1 prediction for now)