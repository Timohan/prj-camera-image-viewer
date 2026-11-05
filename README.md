# prj-camera-image-viewer
360 degree view of the camera images of prj-camera-image-creator. The distance of every pixel is predicted
with the vision transformer that prj-camera-image-learner trained, and the 6 camera images become 3D surfaces
that you can look around with OpenGL (prj-tg-ui-lib).

# Building
Needs prj-tg-ui-lib and OpenCV 4, prj-camera-image-learner is not needed.

    make

The vision transformer that reads the model files is a copy of the sources of prj-camera-image-learner
(src/network and src/global_log.h). If the learner changes them (for example a new model file version), copy them
again and build (the learner must be next to this directory or give its place with LEARNER_PATH):

    make update-learner [LEARNER_PATH=<directory>]
    make clean && make

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
Pixels that the model predicts as sky are put far away (sky dome), so the model must have the sky output
(prj-camera-image-learner model file version 2). Black areas are places that no camera sees.
