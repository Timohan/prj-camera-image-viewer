# prj-camera-image-viewer
360 degree view of the camera images of prj-camera-image-creator. The distance of every pixel is predicted
with the vision transformer that prj-camera-image-learner trained, and the 6 camera images become 3D surfaces
that you can look around with OpenGL (prj-tg-ui-lib).

# Building
Needs prj-tg-ui-lib and OpenCV 4, prj-camera-image-learner is not needed.
OpenCV 4 can be built as told in the README of prj-camera-image-creator.

## prj-tg-ui-lib requirement
The window, the labels and the keyboard and mouse input are from prj-tg-ui-lib
(https://github.com/Timohan/prj-tg-ui-lib.git), it needs prj-ttf-reader (https://github.com/Timohan/prj-ttf-reader)
and the packages libglew-dev, libpng-dev, libglfw3-dev, make and build-essential:

    sudo apt install libglew-dev libpng-dev libglfw3-dev make build-essential
    git clone https://github.com/Timohan/prj-ttf-reader.git
    (build and install prj-ttf-reader as told in its README)
    git clone https://github.com/Timohan/prj-tg-ui-lib.git
    cd prj-tg-ui-lib/lib
    make
    sudo make install

test: pkg-config --modversion prj-tg-ui-lib

## Compiling

    make

The vision transformer that reads the model files and the matching of the cameras are copies of the sources of
prj-camera-image-learner (src/network, src/stereo and src/global_log.h). If the learner changes them (for example a new model file version), copy them
again and build (the learner must be next to this directory or give its place with LEARNER_PATH):

    make update-learner [LEARNER_PATH=<directory>]
    make clean && make

# Opening the created pictures (quick start)
The pictures come from prj-camera-image-creator (see "Creating pictures (quick start)" of its README): key 's'
saves the 6 pictures and data files of one scene, for example
`objects1_20261108-092717_1.png` ... `objects1_20261108-092717_6.png` and their .dat files. The scene name is
the file name without `_<camera number>.png`, here `objects1_20261108-092717`.

1. Build the program:

       make

2. Non-fisheye pictures (cameraobjects_ids.ini of the creator, saved to /mnt/camera/camera_output): open the
   viewer without arguments,

       ./prj-camera-image-viewer

   it uses /mnt/camera/camera_output (cameraImageOutputDirectory=/mnt/camera of configs/general.ini of this
   repository), the scene with the largest objects number (objects10_... if objects1.dat ... objects10.dat were
   saved) and the model /mnt/camera/camera_model/vision_transformer.model, or model/default.model if
   /mnt/camera has no model (model/default.model is the model of the IDS cameras). The used scene and model are
   printed.
3. Fisheye pictures (cameraobjects_elp.ini of the creator, saved to /mnt/camera/fisheye/camera_output): give the
   directory and the fisheye model, the IDS model cannot predict fisheye pictures:

       ./prj-camera-image-viewer --imageDirectory=/mnt/camera/fisheye/camera_output --modelFile=default_fisheye.model

   default_fisheye.model is the model of the ELP fisheye cameras of this repository.
4. Another scene: give its name with --scene, for example objects5 of the fisheye pictures:

       ./prj-camera-image-viewer --imageDirectory=/mnt/camera/fisheye/camera_output --modelFile=default_fisheye.model --scene=objects5_20261108-092801

   (`ls /mnt/camera/fisheye/camera_output` shows the names). One window shows one scene, close the viewer and
   open it again for another scene.
5. In the window: the arrow keys turn the view, the mouse wheel moves it, keys 1-6 move to a camera and 0 to the
   middle of the cameras, 't' changes between the predicted and the correct distances (the .dat files) and a left
   click shows the distance of a point (see Keys below).

The model must be of the same cameras as the pictures (positions and lenses), the viewer matches the 6 pictures
with the cameras of the model before the prediction.

# Usage
    prj-camera-image-viewer [--imageDirectory=<directory>] [--scene=<name>] [--modelFile=<file>]

- `--imageDirectory` images of prj-camera-image-creator, default is `<cameraImageOutputDirectory>/camera_output`
  (cameraImageOutputDirectory of configs/general.ini of this repository)
- `--scene` images `<name>_1.png` ... `<name>_6.png` (and their .dat files, see below), for example
  `--scene=objects999_20261025-175558`, default is the scene that has the largest objects number
- `--modelFile` model of prj-camera-image-learner, without it the model is
  `<cameraImageOutputDirectory>/camera_model/vision_transformer.model`, or if it is missing, the newest .model file of
  `<cameraImageOutputDirectory>/camera_model`, or if there is none, model/default.model of this repository (the used
  file is printed). cameraImageOutputDirectory is in configs/general.ini of this repository (/mnt/camera), if it is
  missing, model/default.model is used.

Keys:
- left/right arrow turn the view, up/down arrow turn it up or down
- mouse wheel moves forward or backward
- 0 moves to the middle of the cameras, 1-6 to the place of the camera
- t changes between the distances predicted by the model and the correct distances of the .dat files (only if the
  .dat files are there), the label at the top left corner shows which ones are shown
- f fills the black areas (places that no camera sees, behind the edges of the objects) with the background behind
  the edge, or shows them black, the label at the top left corner shows if it is on or off (default)
- left click on a point shows the point (window pixels from the top left corner) and the distance of the clicked point
  of the surface from the camera of the image there (`Camera <number> <x>/<y> Distance from camera: <mm>`) with the
  shown distances, `Sky` for the sky and `N/A` for a black area or a pixel that sees nothing. The point is between the
  pixels of its triangle, near the horizon one pixel can be tens of meters long.

The camera positions and lenses are from the model (prj-camera-image-learner model file version 7, the training saves
them), so only the png files are needed, the .dat files are read only for the correct distances if they are there.

Every pixel is put to the camera position + distance along its ray through the lens (lens distortion of the camera
is used). Neighbouring pixels are joined to triangles, except over the edges of the surfaces: the correct
edges are from the surface numbers of the .dat files and the predicted edges from the model, if it has the surface
output (prj-camera-image-learner model file version 4). Without them a jump of the distance is an edge.
A model that is trained with the distances from the other cameras (`prj-camera-image-learner --geo`, model file
version 8) needs them also here: before the prediction the 6 images are matched with the cameras of the model exactly
like the .geo files of the training (StereoMatcher, a few seconds with all CPU threads). model/default.model is such
a model, so the 6 images of the scene must be from the cameras of the model.
Pixels that the model predicts as sky are put far away (sky dome), so the model must have the sky output
(prj-camera-image-learner model file version 2). Black areas are places that no camera sees.
