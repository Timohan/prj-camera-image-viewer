# prj-camera-image-viewer
360 degree view of the camera images of prj-camera-image-creator. The distance of every pixel is predicted
with the vision transformer that prj-camera-image-learner trained, and the 6 camera images become 3D surfaces
that you can look around with OpenGL (prj-tg-ui-lib).

# Building
Needs prj-tg-ui-lib, OpenCV 4 and the sources of prj-camera-image-learner next to this directory
(the vision transformer is compiled from them, another place can be given with `make LEARNER_PATH=<directory>`).

    make

# Usage
    prj-camera-image-viewer [--imageDirectory=<directory>] [--scene=<name>] [--modelFile=<file>]

- `--imageDirectory` images of prj-camera-image-creator, default is /home/timo/camera_output
- `--scene` images `<name>_1.png` ... `<name>_6.png` and their .dat files, for example
  `--scene=objects999_20261025-175558`, default is the scene that has the largest objects number
- `--modelFile` model of prj-camera-image-learner, default is /home/timo/camera_model/vision_transformer.model

Keys:
- left/right arrow turn the view, up/down arrow turn it up or down
- mouse wheel moves forward or backward
- 0 moves to the middle of the cameras, 1-6 to the place of the camera
- t changes between the distances predicted by the model and the correct distances of the .dat files,
  the label at the top left corner shows which ones are shown
- f fills the black areas (places that no camera sees, behind the edges of the objects) with the background behind
  the edge, or shows them black, the label at the top left corner shows if it is on (default) or off
- left click on a point shows the point (window pixels from the top left corner) and the distance of the camera image
  pixel there from its camera (`Camera <number> <x>/<y> Distance from camera: <mm>`) with the shown distances, `Sky`
  for the sky and `N/A` for a black area or a pixel that sees nothing

Every pixel is put to the camera position + distance along its ray through the lens (lens distortion of the .dat
file is used). Neighbouring pixels are joined to triangles, except over the edges of the surfaces: the correct
edges are from the surface numbers of the .dat files and the predicted edges from the model, if it has the surface
output (prj-camera-image-learner model file version 4). Without them a jump of the distance is an edge.
Pixels that the model predicts as sky are put far away (sky dome), so the model must have the sky output
(prj-camera-image-learner model file version 2). Black areas are places that no camera sees.
