> STATUS: Discontinued.
> 
> This project was made part of our Master's degree at EPITA. The project is now discontinued
> and will not receive any improvement / bug fix.

# Pathtracer with CUDA!

<p align="center">
  <img width="420" src="https://user-images.githubusercontent.com/8783766/33244598-ec130f1c-d2fa-11e7-89fd-ea1a108c8802.png">
  <img width="420" src="https://user-images.githubusercontent.com/8783766/33244599-ee4330f0-d2fa-11e7-9fa1-06a8b8215f7a.png">
</p>

Simple Pathtracer made for our CUDA course at EPITA, a French Computer Science school.

## Features

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/8783766/33244854-f8c8cafe-d2fe-11e7-982c-17393e29f634.gif">
</p>

### Scenes

In order to load a scene (or several at a time), you have to provide them on the command line.
A scene is a special ASCII file having the following syntax:

```
cubemap cubemap/garden.jpg

# camera x y z dir_x dir_y dir_z fov_x dof_focus dof_aperture
camera -2.7 2.06 2.52 0.62 -0.348 -0.7 90.0 3.555 0.01

# pos_x pos_y pos_z r g b emission radius
p_light 2.9 2.1 2.9 1.0 1.0 1.0 2.0 0.3

# pos_x pos_y pos_z r g b emission radius
p_light -2.7 2.1 -2.55 1.0 1.0 1.0 2.0 0.3

# pos_x pos_y pos_z r g b emission radius
p_light 2.9 2.1 -2.55 1.0 1.0 1.0 2.0 0.3

scene obj/indoor.obj
```

Inside it, you have to reference the obj file you want to use, as well as a cubemap,
some lights and a camera with some initial values.

### Textures

We support the following textures:
* Diffuse
* Normal
* Specular
* Cubemaps

### Algoritm

Our algorithm works using few samples, by using temporal buffering.

## Build

### Dependencies

You will need the following dependencies (fetched if using our CMake):
* [GLFW](http://www.glfw.org/)

### Windows

You will have to load the .sln, and go to the project properties window, in order to change the link to your include folders as well as the dependencies.

### Linux

You can build the solution using:

```sh
sh$ mkdir build && cd build && cmake .. && make
```

## Tool to visualize the sampling of our BRDF

At the root of the project, you can find a python Jupyter Notebook that helps visualize our BRDF sampling depending on roughness and such.
