#ifndef SUB_SURFACE_H
#define SUB_SURFACE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/*!
 * Sub-surfaces of the objects of prj-camera-image-creator: the parts of an object that have one geometric form.
 * - side wall: a flat face of a cube (6 of them), its corners are the corners of the face above the floor, a face that
 *   goes under the floor has the cut corners on the floor (3-5 corners),
 * - curved wall: the side of a cylinder (an elliptic cylinder: axis, the radii of the ellipse and the height),
 * - cone wall: the side of a cone (its base is a round wall),
 * - round wall: a flat ellipse, an end of a cylinder or the base of a cone (center, normal, radii), the floor can cut it,
 * - ball: an ellipsoid (center, the 3 radii and their directions), the floor can cut it,
 * - floor: the ground (y = 0),
 * - free-form: a car or a person, its box is the object box of the data file.
 * The true sub-surfaces come from the data files (the surface number and the distance of every pixel, the box of
 * every object in "OBJS") and the objects file of the scene (the shape of every cube, ball, cylinder and cone): the
 * point of a pixel in the coordinates of its object tells the face it is on.
 * The visible part of a sub-surface is measured from its pixels: the angles around the axis (0 is +z of the object,
 * positive turns to +x) and the heights along the axis that the cameras see. Several images (cameras) of a scene can be
 * merged, a sub-surface has the same id in all of them.
 * World coordinates are the ones of the data files: x right, y up, z forward (mm).
 */
namespace subsurface {

enum Type : uint8_t {
    TypeNone = 0,       // sky and the pixels that see nothing
    TypeFloor,
    TypeSideWall,
    TypeCurvedWall,
    TypeConeWall,
    TypeRoundWall,
    TypeBall,
    TypeFreeForm,
    TYPE_COUNT
};

// shapes of the objects file
enum Shape : uint8_t {
    ShapeNone = 0,      // sky, ground or not known
    ShapeCube,
    ShapeBall,
    ShapeCylinder,
    ShapeCone,
    ShapeCar,
    ShapePerson
};

// surface numbers of the data files: 0 sky, 1 floor, then the objects of the objects file
const uint32_t SURFACE_FLOOR = 1;
const uint32_t SURFACE_FIRST_OBJECT = 2;
// the id of a sub-surface is surface number * SLOTS + its index in the object
const uint32_t SLOTS = 8;
const int ANGLE_BINS = 360;
// mm: a point this near the floor is on it
const double FLOOR_TOLERANCE = 10.0;

// object of a surface number: the class and the box of the data file ("OBJS"), the shape of the objects file
struct Object {
    uint32_t objectClass = 0;       // class of the data file: 0 sky, 1 ground, 2 unknown object, 3 sedan ... 8 child
    Shape shape = ShapeNone;
    double center[3] = { 0.0, 0.0, 0.0 };   // mm
    double size[3] = { 0.0, 0.0, 0.0 };     // mm
    double yaw = 0.0;                       // degrees, rotated roll first, then pitch and yaw (like the creator)
    double pitch = 0.0;
    double roll = 0.0;
};

struct Corner {
    double point[3];    // mm
    bool cut;           // made by the floor (the object goes under it), not a corner of the whole face
};

struct Part {
    uint32_t id = 0;            // surface * SLOTS + index
    uint32_t surface = 0;
    int index = 0;              // cube: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z; cylinder: 0 side, 1 top (+y), 2 bottom;
                                // cone: 0 side, 1 base
    Type type = TypeNone;
    Shape shape = ShapeNone;
    uint32_t objectClass = 0;
    // the object: center, world directions of its x, y and z axes and its half sizes (mm)
    double objectCenter[3] = { 0.0, 0.0, 0.0 };
    double axes[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    double half[3] = { 0.0, 0.0, 0.0 };
    // side wall and round wall: center and outward normal of the face; curved wall, cone wall, ball: the center of the
    // object and its y axis (cone: the center of the base and the axis toward the tip)
    double center[3] = { 0.0, 0.0, 0.0 };
    double normal[3] = { 0.0, 1.0, 0.0 };
    // side wall: corners of the face above the floor in order around it; round wall: the ends of the floor cut;
    // curved and cone wall: where the visible part begins (bottom, top) and ends (bottom, top)
    std::vector<Corner> corners;
    bool floorCut = false;      // the floor cuts the sub-surface (a part of it is under the floor)
    // visible part (pixels of the images), see finish()
    size_t pixels = 0;
    double residualSum = 0.0;   // mm, distance of the pixel points from the sub-surface
    double residualMax = 0.0;
    std::vector<float> binLow;  // lowest and highest y (mm, object axes) of the pixels in every angle bin, NaN: empty
    std::vector<float> binHigh;
    double lowY = 0.0;          // lowest and highest y of all pixels (mm, object axes)
    double highY = 0.0;
    bool fullCircle = false;    // the pixels are all around the axis
    double angleBegin = 0.0;    // degrees [0, 360), visible part from angleBegin to angleEnd (may go over 360)
    double angleEnd = 0.0;
};

const char *typeName(Type type);
const char *shapeName(Shape shape);

// "<objects file directory>/<objects file>" of an image or data file of the creator:
// <dir>/camera_output/objects17525_20261114-060631_3.dat -> <dir>/camera_objects/objects17525.dat
std::string objectsFileOf(const std::string &imageFile);
// sets the shapes of the objects (by surface number) from the objects file: the cubes, balls, cylinders and cones are
// the first objects in this order (Cube=, Ball=, Cylinder=, Cone= lines), the cars and the people come from the class,
// false (and the reason) if the file cannot be read or its objects are not the ones of the data file
bool setShapes(const std::string &objectsFile, std::vector<Object> *objects, std::string *error);

// sub-surface of every pixel of an image: points are the world points of the pixels (3 values, mm), surfaces the
// surface numbers (0 sky); partOfPixel gets the index of the part of every pixel (-1: none), parts the sub-surfaces
// (finished)
void label(const double *points, const uint32_t *surfaces, size_t count, const std::vector<Object> &objects,
           std::vector<int32_t> *partOfPixel, std::vector<Part> *parts);
// adds the parts of an image to the parts of the scene (the same id is the same sub-surface) and finishes them
void merge(std::vector<Part> *sceneParts, const std::vector<Part> &imageParts);
// visible angles and corners from the pixels
void finish(Part *part);

// ---- predicted sub-surfaces (the sub-surface outputs of a model, pixel_label.h) and their fit

// a predicted sub-surface: pixels of the same predicted type whose ids are near each other
struct PredictedPart {
    int type = -1;                  // 0..SUB_TYPE_COUNT - 1 (Type - 1)
    std::vector<size_t> pixels;
    double id[8] = {};              // means of the outputs of the pixels (SUB_ID_SIZE)
    double axis[3] = {};            // the axes are turned to the same direction before the mean, unit length
    double center[3] = {};          // m (the outputs are in OBJECT_CENTER_SCALE m)
};
// joins the pixels that have use[i] != 0 (values: SUB_SURFACE_OUTPUT_COUNT outputs of every pixel, width x height) to
// predicted sub-surfaces: a pixel joins the sub-surface of its type whose mean id is nearest if it is nearer than
// groupDistance, otherwise it starts a new one; the sub-surfaces of fewer than minPixels pixels are dropped,
// partOfPixel gets the index of the sub-surface of every pixel (-1: none)
void groupPixels(const float *values, const uint8_t *use, int width, int height, double groupDistance, size_t minPixels,
                 std::vector<PredictedPart> *parts, std::vector<int32_t> *partOfPixel);

// geometry of the pixels of a sub-surface from their points (the same fit for the predicted and the true pixels):
// a side wall, round wall or floor a plane (least squares), a side wall also the corners of its visible part (the
// convex outline of its points on the plane without the corners that cut less than SIMPLIFY_AREA of its area); a curved
// wall or a cone wall the visible angle around the axis (through center) and where the visible part begins and ends
// (the lowest and highest point along the axis of the first and last angles)
struct PartFit {
    bool valid = false;
    double normal[3] = { 0.0, 0.0, 0.0 };
    std::vector<std::vector<double>> corners;   // 3 values each, in the units of the points
    double span = 0.0;                          // degrees
};
PartFit fitPart(int type, const std::vector<size_t> &pixels, const double *points, const double *axis, const double *center);
// mean distance of the corners of a from the nearest corner of b and of b from a (in the units of the points),
// negative if either has none
double cornerDistance(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b);

// one line description, lengths in m, for example "side wall 3 of a cube (surface 12): 5 corners, 2 cut by the floor"
std::string summary(const Part &part);
// the geometry, lengths in m: the corners (side wall), where the visible part begins and ends (curved and cone wall),
// the floor cut (round wall), the center, the normal or the axis and the radius directions, the visible angles
// and heights
std::string geometry(const Part &part);
// the id, summary(), the pixels and their mean distance from the sub-surface, geometry()
std::string describe(const Part &part);

}

#endif // SUB_SURFACE_H
