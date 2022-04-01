# Active Exploration for Neural Global Illumination of Variable Scenes
#### Stavros Diolatzis, Julien Philip, George Drettakis
##### ACM Transactions on Graphics 2022

![Teaser Image](http://www-sop.inria.fr/reves/Basilic/2022/DPD22/teaser.jpg)

#### [Project Website](https://repo-sam.inria.fr/fungraph/active-exploration/)

Abstract: *Neural rendering algorithms introduce a fundamentally new approach for photorealistic rendering, typically by learning a neural representation of illumination on large numbers of ground truth images. When training for a given variable scene, i.e., changing objects, materials, lights and viewpoint, the space D of possible training data instances quickly becomes unmanageable as the dimensions of variable parameters increase. We introduce a novel Active Exploration method using Markov Chain Monte Carlo, which explores D , generating samples (i.e., ground truth renderings) that best help training and interleaves training and on-the-fly sample data generation. We introduce a self-tuning sample reuse strategy to minimize the expensive step of rendering training samples. We apply our approach on a neural generator that learns to render novel scene instances given an explicit parameterization of the scene configuration. Our results show that Active Exploration trains our network much more efficiently than uniformly sampling, and together with our resolution enhancement approach, achieves better quality than uniform sampling at convergence. Our method allows interactive rendering of hard light transport paths (e.g., complex caustics) – that require very high samples counts to be captured – and provides dynamic scene navigation and manipulation, after training for 5-18 hours depending on required quality and variations.*

## Requirements

* 64-bit Python 3.8 and Pytorch 1.8.1 or later. Preferably use Miniconda for Python if you are having issues with Mitsuba 2.
* CUDA Toolkit 10.2 or later.
* For Python Libraries see environment.yml

## Compiling and Running

1. Clone the repository with `--recursive` to get all the dependencies.
2. Compile Mitsuba 2, following the [documentation](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling.html) make sure to have all the prerequisites. 
3. If you are having issues with the Python scripts not finding the mitsuba/enoki module look at the Mitsuba 2 documentation above and at their [github](https://github.com/mitsuba-renderer/mitsuba2) 

You are ready to go!

## Pretrained Models

The pretrained models are here: [Models](./models/)

To interactively preview the neural rendering using the pretraine models run the `preview_widget_varying.py` script with the following arguments:

```
--scene_path <path-to-preview.xml> 
--scene_buffers_path <path-to-preview_buffers.xml>
--arch ppixel --hidden_features 512 --hidden_layers 8 --tonemap log1p 
--model_path <path-to-model.pth> 
--metric l1
```

## Training a new model

To train a new model run the `train_dynamic_markov_reuse.py` script for one of the scenes. An example is:

```
train_dynamic_markov_reuse_grad_res.py --scene_path <path-to-scene.xml> --models_dir <path-to-store-model> --training_samples 2000 --validation_samples 0 --tonemap log1p --hidden_features 512 --hidden_layers 8 --reuse_bias 4.6 --ema_alpha 0.9 --arch ppixel --patch_size 32 --num_patches 16 --num_threads 16 --timeout 25 --max_res 600
```

The scenes used in the paper are provided in: [Scenes](./scenes/). These are variations of some of the scenes from [Bitterli's repository](https://benedikt-bitterli.me/resources/). We are thankful to him for providing them.

## Defining New Variable Scenes

Our method includes a custom version of Mitsuba 2 that allows users to define variable parts of a scene through the scene's xml file. It is easy to use and once defined and the model is trained the users can interact with the variable elements.

Each scene needs to have *three* xml files (Look into the provided scenes for examples).
* `scene.xml` -- This file is the scene definition used during training. It needs to have low resolution (proposed resolution: 128x128) and to render both the image and the buffers by using the `aov` integrator. It also needs to have enough spp for the required effects to appear (proposed spp: 400-2000). It is not a problem if the result is noisy since the model will learn to denoise in world space anyways.
* `preview_buffers.xml` -- Once the model is trained this xml is used to render the buffers in high resolution for inference. This xml should have the target resolution (proposed resolution: 600x600) and render only the buffers with the `aov` integrator. The spp here are only used for antialiasing and so they can be very low (proposed spp: 20).
* `preview.xml` -- This is the xml used to render the ground truth at the target resolution. It should have the same resolution as the `preview_buffers.xml` and high spp.

We suggest that you first build your `scene.xml`, add variations and then create the `preview_buffers.xml` and `preview.xml` by adjusting the spp and resolution.

In order to introduce variability into a scene you will need to edit the xml following the guidelines in [Variations](./docs/variations.md)

