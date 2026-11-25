#include "sub_surface.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

namespace subsurface {

// mm: the center of an object of the objects file must be this near the center of the data file
#define OBJECT_MATCH_TOLERANCE  1.0
// the begin and end lines of a curved wall: the heights of the pixels of this share of the visible angles at its ends
#define EDGE_SHARE              0.05
#define EDGE_MIN_BINS           3

static const char *const TYPE_NAMES[TYPE_COUNT] = { "none", "floor", "side wall", "curved wall", "cone wall",
                                                    "round wall", "ball", "free-form" };
static const char *const SHAPE_NAMES[] = { "none", "cube", "ball", "cylinder", "cone", "car", "person" };

const char *typeName(Type type)
{
    return type < TYPE_COUNT ? TYPE_NAMES[type] : "unknown";
}

const char *shapeName(Shape shape)
{
    return shape <= ShapePerson ? SHAPE_NAMES[shape] : "unknown";
}

std::string objectsFileOf(const std::string &imageFile)
{
    const std::filesystem::path path(imageFile);
    std::string name = path.filename().string();
    for (const char *extension : { ".dat", ".png", ".geo" }) {
        const size_t length = strlen(extension);
        if (name.size() > length && name.compare(name.size() - length, length, extension) == 0) {
            name.resize(name.size() - length);
        }
    }
    // <objects file name>_<timestamp>_<camera number>, the scene name has no camera number
    for (int part=0;part<2;part++) {
        const size_t underscore = name.rfind('_');
        if (underscore == std::string::npos || underscore == 0) {
            break;
        }
        const std::string last = name.substr(underscore + 1);
        // the timestamp has a '-', the camera number only digits
        const bool cameraNumber = !last.empty() && last.find_first_not_of("0123456789") == std::string::npos;
        const bool timestamp = last.size() == 15 && last[8] == '-';
        if (!cameraNumber && !timestamp) {
            break;
        }
        name.resize(underscore);
        if (timestamp) {
            break;
        }
    }
    return (path.parent_path().parent_path()/"camera_objects"/(name + ".dat")).string();
}

bool setShapes(const std::string &objectsFile, std::vector<Object> *objects, std::string *error)
{
    std::ifstream file(objectsFile);
    if (!file.is_open()) {
        *error = "cannot read " + objectsFile;
        return false;
    }
    // the creator gives the surface numbers in this order of the keys, every key in the order of the file
    const std::pair<const char *, Shape> keys[] = { { "Cube", ShapeCube }, { "Ball", ShapeBall },
                                                    { "Cylinder", ShapeCylinder }, { "Cone", ShapeCone } };
    std::map<std::string, std::vector<std::vector<double>>> values;
    std::string line;
    while (std::getline(file, line)) {
        const size_t equal = line.find('=');
        if (line.empty() || line[0] == '#' || equal == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, equal);
        std::vector<double> numbers;
        std::stringstream stream(line.substr(equal + 1));
        std::string item;
        while (std::getline(stream, item, '/')) {
            char *end = nullptr;
            const double number = strtod(item.c_str(), &end);
            if (end == item.c_str() || *end != '\0') {
                break;      // the texture
            }
            numbers.push_back(number);
        }
        values[key].push_back(numbers);
    }
    uint32_t surface = SURFACE_FIRST_OBJECT;
    for (const auto &[key, shape] : keys) {
        for (const std::vector<double> &numbers : values[key]) {
            if (numbers.size() < 9) {
                *error = objectsFile + ": " + key + " has too few values";
                return false;
            }
            if (surface >= objects->size() || (*objects)[surface].objectClass != 2) {
                *error = objectsFile + ": " + key + " " + std::to_string(surface - SURFACE_FIRST_OBJECT + 1)
                         + " is not an unknown object (surface " + std::to_string(surface) + ") of the data file";
                return false;
            }
            Object &object = (*objects)[surface];
            for (int i=0;i<3;i++) {
                if (std::fabs(object.center[i] - numbers[static_cast<size_t>(i)]) > OBJECT_MATCH_TOLERANCE) {
                    *error = objectsFile + ": " + key + " at surface " + std::to_string(surface)
                             + " is not at the center of the data file";
                    return false;
                }
            }
            object.shape = shape;
            surface++;
        }
    }
    for (size_t i=0;i<objects->size();i++) {
        Object &object = (*objects)[i];
        if (object.objectClass >= 3 && object.objectClass <= 5) {
            object.shape = ShapeCar;
        } else if (object.objectClass >= 6 && object.objectClass <= 8) {
            object.shape = ShapePerson;
        } else if (object.objectClass == 2 && object.shape == ShapeNone) {
            *error = objectsFile + " has no shape for surface " + std::to_string(i);
            return false;
        }
    }
    return true;
}

// world direction of a vector of the object: roll first, then pitch and yaw (Cube::init() of the creator)
static void rotate(const Object &object, const double in[3], double out[3])
{
    const double yaw = object.yaw*M_PI/180.0, pitch = object.pitch*M_PI/180.0, roll = object.roll*M_PI/180.0;
    const double rollX = in[0]*std::cos(roll) + in[1]*std::sin(roll);
    const double rollY = -in[0]*std::sin(roll) + in[1]*std::cos(roll);
    const double pitchY = rollY*std::cos(pitch) + in[2]*std::sin(pitch);
    const double pitchZ = -rollY*std::sin(pitch) + in[2]*std::cos(pitch);
    out[0] = rollX*std::cos(yaw) + pitchZ*std::sin(yaw);
    out[1] = pitchY;
    out[2] = -rollX*std::sin(yaw) + pitchZ*std::cos(yaw);
}

// world point of the object coordinates (mm)
static void toWorld(const Part &part, const double local[3], double world[3])
{
    for (int k=0;k<3;k++) {
        world[k] = part.objectCenter[k] + local[0]*part.axes[0][k] + local[1]*part.axes[1][k] + local[2]*part.axes[2][k];
    }
}

static void toLocal(const Part &part, const double *world, double local[3])
{
    const double d[3] = { world[0] - part.objectCenter[0], world[1] - part.objectCenter[1],
                          world[2] - part.objectCenter[2] };
    for (int i=0;i<3;i++) {
        local[i] = d[0]*part.axes[i][0] + d[1]*part.axes[i][1] + d[2]*part.axes[i][2];
    }
}

// lowest world y of the ellipse of the object at the height y (object axes), mm
static double ellipseLowest(const Part &part, double y, double scale)
{
    const double center = part.objectCenter[1] + y*part.axes[1][1];
    const double a = scale*part.half[0]*part.axes[0][1], b = scale*part.half[2]*part.axes[2][1];
    return center - std::sqrt(a*a + b*b);
}

// world point of the side of a cylinder (scale 1) or a cone at the angle (degrees around y, 0 is +z) and the height y
static void wallPoint(const Part &part, double angle, double y, double world[3])
{
    double scale = 1.0;
    if (part.type == TypeConeWall) {
        scale = part.half[1] > 0.0 ? std::max(0.0, (part.half[1] - y)/(2.0*part.half[1])) : 0.0;
    }
    const double radians = angle*M_PI/180.0;
    const double local[3] = { scale*part.half[0]*std::sin(radians), y, scale*part.half[2]*std::cos(radians) };
    toWorld(part, local, world);
}

// the corners of a face above the floor: polygon clipped by y >= 0, the new corners are cut
static std::vector<Corner> clipByFloor(const std::vector<Corner> &polygon)
{
    std::vector<Corner> result;
    for (size_t i=0;i<polygon.size();i++) {
        const Corner &a = polygon[i];
        const Corner &b = polygon[(i + 1)%polygon.size()];
        const bool aInside = a.point[1] >= 0.0, bInside = b.point[1] >= 0.0;
        if (aInside) {
            result.push_back(a);
        }
        if (aInside != bInside) {
            const double t = a.point[1]/(a.point[1] - b.point[1]);
            Corner cut;
            for (int k=0;k<3;k++) {
                cut.point[k] = a.point[k] + t*(b.point[k] - a.point[k]);
            }
            cut.point[1] = 0.0;
            cut.cut = true;
            result.push_back(cut);
        }
    }
    return result;
}

// the geometry of a new sub-surface of the object
static void initPart(Part *part, const Object &object, uint32_t surface, int index)
{
    part->id = surface*SLOTS + static_cast<uint32_t>(index);
    part->surface = surface;
    part->index = index;
    part->shape = object.shape;
    part->objectClass = object.objectClass;
    part->binLow.clear();
    part->binHigh.clear();
    part->lowY = std::numeric_limits<double>::infinity();
    part->highY = -std::numeric_limits<double>::infinity();
    if (surface == SURFACE_FLOOR) {
        part->type = TypeFloor;
        return;
    }
    for (int i=0;i<3;i++) {
        part->objectCenter[i] = object.center[i];
        part->half[i] = object.size[i]/2.0;
        const double unit[3] = { i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0 };
        rotate(object, unit, part->axes[i]);
    }
    const double *X = part->axes[0], *Y = part->axes[1], *Z = part->axes[2];
    const double *h = part->half;
    std::copy(object.center, object.center + 3, part->center);
    std::copy(Y, Y + 3, part->normal);
    switch (object.shape) {
    case ShapeCube: {
        part->type = TypeSideWall;
        const int axis = index/2;
        const double sign = index%2 == 0 ? 1.0 : -1.0;
        const int b = (axis + 1)%3, c = (axis + 2)%3;
        for (int k=0;k<3;k++) {
            part->normal[k] = sign*part->axes[axis][k];
            part->center[k] = object.center[k] + sign*h[axis]*part->axes[axis][k];
        }
        // corners around the face
        const double signs[4][2] = { { -1.0, -1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 }, { -1.0, 1.0 } };
        std::vector<Corner> polygon;
        for (const auto &s : signs) {
            double local[3];
            local[axis] = sign*h[axis];
            local[b] = s[0]*h[b];
            local[c] = s[1]*h[c];
            Corner corner;
            toWorld(*part, local, corner.point);
            corner.cut = false;
            polygon.push_back(corner);
        }
        part->corners = clipByFloor(polygon);
        part->floorCut = std::any_of(part->corners.begin(), part->corners.end(), [](const Corner &corner) {
            return corner.cut;
        });
        break;
    }
    case ShapeCylinder:
    case ShapeCone:
        if (index == 0) {
            part->type = object.shape == ShapeCylinder ? TypeCurvedWall : TypeConeWall;
            if (object.shape == ShapeCylinder) {
                part->floorCut = std::min(ellipseLowest(*part, h[1], 1.0), ellipseLowest(*part, -h[1], 1.0))
                                 < -FLOOR_TOLERANCE;
            } else {
                for (int k=0;k<3;k++) {
                    part->center[k] = object.center[k] - h[1]*Y[k];
                }
                part->floorCut = std::min(ellipseLowest(*part, -h[1], 1.0), object.center[1] + h[1]*Y[1])
                                 < -FLOOR_TOLERANCE;
            }
        } else {
            // cylinder: 1 top, 2 bottom; cone: 1 base (bottom)
            part->type = TypeRoundWall;
            const double sign = object.shape == ShapeCylinder && index == 1 ? 1.0 : -1.0;
            for (int k=0;k<3;k++) {
                part->center[k] = object.center[k] + sign*h[1]*Y[k];
                part->normal[k] = sign*Y[k];
            }
            // floor cut of the ellipse: center.y + h[2]*cos(a)*Z.y + h[0]*sin(a)*X.y = 0
            const double A = h[2]*Z[1], B = h[0]*X[1];
            const double R = std::sqrt(A*A + B*B);
            if (R > 0.0 && std::fabs(part->center[1]) < R) {
                const double base = std::atan2(B, A), spread = std::acos(-part->center[1]/R);
                for (double angle : { base - spread, base + spread }) {
                    Corner corner;
                    const double local[3] = { h[0]*std::sin(angle), sign*h[1], h[2]*std::cos(angle) };
                    toWorld(*part, local, corner.point);
                    corner.point[1] = 0.0;
                    corner.cut = true;
                    part->corners.push_back(corner);
                }
                part->floorCut = true;
            } else {
                part->floorCut = part->center[1] < 0.0;
            }
        }
        break;
    case ShapeBall: {
        part->type = TypeBall;
        const double a = h[0]*X[1], b = h[1]*Y[1], c = h[2]*Z[1];
        part->floorCut = object.center[1] - std::sqrt(a*a + b*b + c*c) < -FLOOR_TOLERANCE;
        break;
    }
    default:
        part->type = TypeFreeForm;
        part->floorCut = false;
        break;
    }
    if (part->type == TypeCurvedWall || part->type == TypeConeWall || part->type == TypeRoundWall
        || part->type == TypeBall) {
        part->binLow.assign(ANGLE_BINS, std::numeric_limits<float>::quiet_NaN());
        part->binHigh.assign(ANGLE_BINS, std::numeric_limits<float>::quiet_NaN());
    }
}

// face of the object that the point is on, and the distance of the point from it (mm)
static int faceOf(const Object &object, const double local[3], double *residual)
{
    const double hx = object.size[0]/2.0, hy = object.size[1]/2.0, hz = object.size[2]/2.0;
    const double minRadius = std::min(hx, hz);
    switch (object.shape) {
    case ShapeCube: {
        int best = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        const double h[3] = { hx, hy, hz };
        for (int i=0;i<3;i++) {
            const double distance = std::fabs(h[i] - std::fabs(local[i]));
            if (distance < bestDistance) {
                bestDistance = distance;
                best = 2*i + (local[i] >= 0.0 ? 0 : 1);
            }
        }
        *residual = bestDistance;
        return best;
    }
    case ShapeCylinder: {
        const double radius = std::sqrt(local[0]*local[0]/(hx*hx) + local[2]*local[2]/(hz*hz));
        const double side = std::fabs(radius - 1.0)*minRadius;
        const double top = std::fabs(hy - local[1]), bottom = std::fabs(hy + local[1]);
        if (side <= top && side <= bottom) {
            *residual = side;
            return 0;
        }
        *residual = std::min(top, bottom);
        return top <= bottom ? 1 : 2;
    }
    case ShapeCone: {
        const double radius = std::sqrt(local[0]*local[0]/(hx*hx) + local[2]*local[2]/(hz*hz));
        const double scale = (hy - local[1])/(2.0*hy);
        const double side = std::fabs(radius - scale)*minRadius;
        const double base = std::fabs(hy + local[1]);
        *residual = std::min(side, base);
        return side <= base ? 0 : 1;
    }
    case ShapeBall: {
        const double radius = std::sqrt(local[0]*local[0]/(hx*hx) + local[1]*local[1]/(hy*hy)
                                        + local[2]*local[2]/(hz*hz));
        *residual = std::fabs(radius - 1.0)*std::min(minRadius, hy);
        return 0;
    }
    default:
        *residual = 0.0;
        return 0;
    }
}

void label(const double *points, const uint32_t *surfaces, size_t count, const std::vector<Object> &objects,
           std::vector<int32_t> *partOfPixel, std::vector<Part> *parts)
{
    partOfPixel->assign(count, -1);
    parts->clear();
    std::unordered_map<uint32_t, int32_t> partOfId;
    std::unordered_map<uint32_t, Part> frames;  // axes of the objects
    const Object floor;
    for (size_t i=0;i<count;i++) {
        const uint32_t surface = surfaces[i];
        const double *point = points + 3*i;
        if (surface == 0 || !std::isfinite(point[0])) {
            continue;
        }
        const Object *object = nullptr;
        if (surface == SURFACE_FLOOR) {
            object = &floor;
        } else if (surface < objects.size() && objects[surface].shape != ShapeNone) {
            object = &objects[surface];
        } else {
            continue;
        }
        int index = 0;
        double residual = 0.0;
        double local[3] = { 0.0, 0.0, 0.0 };
        if (surface == SURFACE_FLOOR) {
            residual = std::fabs(point[1]);
        } else {
            auto frame = frames.find(surface);
            if (frame == frames.end()) {
                Part part;
                initPart(&part, *object, surface, 0);
                frame = frames.emplace(surface, part).first;
            }
            toLocal(frame->second, point, local);
            index = faceOf(*object, local, &residual);
        }
        const uint32_t id = surface*SLOTS + static_cast<uint32_t>(index);
        auto found = partOfId.find(id);
        if (found == partOfId.end()) {
            Part part;
            initPart(&part, *object, surface, index);
            parts->push_back(part);
            found = partOfId.emplace(id, static_cast<int32_t>(parts->size() - 1)).first;
        }
        Part &part = (*parts)[static_cast<size_t>(found->second)];
        (*partOfPixel)[i] = found->second;
        part.pixels++;
        part.residualSum += residual;
        part.residualMax = std::max(part.residualMax, residual);
        if (surface == SURFACE_FLOOR) {
            continue;
        }
        part.lowY = std::min(part.lowY, local[1]);
        part.highY = std::max(part.highY, local[1]);
        if (!part.binLow.empty()) {
            const double hx = part.half[0], hz = part.half[2];
            double angle = std::atan2(local[0]/hx, local[2]/hz)*180.0/M_PI;
            if (angle < 0.0) {
                angle += 360.0;
            }
            const size_t bin = std::min(static_cast<size_t>(angle*ANGLE_BINS/360.0), static_cast<size_t>(ANGLE_BINS - 1));
            const float y = static_cast<float>(local[1]);
            if (std::isnan(part.binLow[bin]) || y < part.binLow[bin]) {
                part.binLow[bin] = y;
            }
            if (std::isnan(part.binHigh[bin]) || y > part.binHigh[bin]) {
                part.binHigh[bin] = y;
            }
        }
    }
    for (Part &part : *parts) {
        finish(&part);
    }
}

void merge(std::vector<Part> *sceneParts, const std::vector<Part> &imageParts)
{
    for (const Part &image : imageParts) {
        auto found = std::find_if(sceneParts->begin(), sceneParts->end(), [&image](const Part &part) {
            return part.id == image.id;
        });
        if (found == sceneParts->end()) {
            sceneParts->push_back(image);
            continue;
        }
        Part &part = *found;
        part.pixels += image.pixels;
        part.residualSum += image.residualSum;
        part.residualMax = std::max(part.residualMax, image.residualMax);
        part.lowY = std::min(part.lowY, image.lowY);
        part.highY = std::max(part.highY, image.highY);
        for (size_t bin=0;bin<part.binLow.size() && bin<image.binLow.size();bin++) {
            if (!std::isnan(image.binLow[bin]) && (std::isnan(part.binLow[bin]) || image.binLow[bin] < part.binLow[bin])) {
                part.binLow[bin] = image.binLow[bin];
            }
            if (!std::isnan(image.binHigh[bin])
                && (std::isnan(part.binHigh[bin]) || image.binHigh[bin] > part.binHigh[bin])) {
                part.binHigh[bin] = image.binHigh[bin];
            }
        }
    }
    for (Part &part : *sceneParts) {
        finish(&part);
    }
    std::sort(sceneParts->begin(), sceneParts->end(), [](const Part &a, const Part &b) {
        return a.id < b.id;
    });
}

void finish(Part *part)
{
    if (part->binLow.empty()) {
        return;
    }
    // the visible part is around the axis outside of the largest gap of empty angle bins
    int gapStart = -1, gapLength = 0;
    for (int start=0;start<ANGLE_BINS;start++) {
        if (!std::isnan(part->binLow[static_cast<size_t>(start)])
            || std::isnan(part->binLow[static_cast<size_t>((start + ANGLE_BINS - 1)%ANGLE_BINS)]) ) {
            continue;
        }
        int length = 0;
        while (length < ANGLE_BINS && std::isnan(part->binLow[static_cast<size_t>((start + length)%ANGLE_BINS)])) {
            length++;
        }
        if (length > gapLength) {
            gapLength = length;
            gapStart = start;
        }
    }
    const bool empty = std::all_of(part->binLow.begin(), part->binLow.end(), [](float value) {
        return std::isnan(value);
    });
    if (empty) {
        return;
    }
    part->fullCircle = gapLength == 0;
    const int begin = part->fullCircle ? 0 : (gapStart + gapLength)%ANGLE_BINS;
    const int end = part->fullCircle ? ANGLE_BINS - 1 : (gapStart + ANGLE_BINS - 1)%ANGLE_BINS;
    part->angleBegin = begin*360.0/ANGLE_BINS;
    part->angleEnd = (end + 1)*360.0/ANGLE_BINS;
    if (part->angleEnd <= part->angleBegin) {
        part->angleEnd += 360.0;
    }
    if (part->type != TypeCurvedWall && part->type != TypeConeWall) {
        return;
    }
    // where the visible part of the wall begins and ends: the lowest and highest pixel of the first and last bin
    part->corners.clear();
    if (part->fullCircle) {
        return;
    }
    // the silhouette at the first and last bin has only a few pixels, so the heights are the lowest and highest pixel
    // of the first and last EDGE_SHARE of the visible angles (at least EDGE_MIN_BINS bins)
    const int visibleBins = (end - begin + ANGLE_BINS)%ANGLE_BINS + 1;
    const int edgeBins = std::max(EDGE_MIN_BINS, static_cast<int>(std::lround(EDGE_SHARE*visibleBins)));
    for (int bin : { begin, end }) {
        const double angle = (bin + (bin == begin ? 0.0 : 1.0))*360.0/ANGLE_BINS;
        float low = std::numeric_limits<float>::infinity(), high = -std::numeric_limits<float>::infinity();
        for (int step=0;step<edgeBins && step<visibleBins;step++) {
            const size_t edge = static_cast<size_t>((bin + (bin == begin ? step : -step) + ANGLE_BINS)%ANGLE_BINS);
            if (!std::isnan(part->binLow[edge])) {
                low = std::min(low, part->binLow[edge]);
                high = std::max(high, part->binHigh[edge]);
            }
        }
        // the heights of the nearby angles can put the end of the line a little under the floor: the world y is
        // linear along the line, so the end goes to the floor (y = 0)
        double lowPoint[3], highPoint[3];
        wallPoint(*part, angle, low, lowPoint);
        wallPoint(*part, angle, high, highPoint);
        if (lowPoint[1] < 0.0 && highPoint[1] > 0.0) {
            low += static_cast<float>(lowPoint[1]/(lowPoint[1] - highPoint[1]))*(high - low);
        } else if (highPoint[1] < 0.0 && lowPoint[1] > 0.0) {
            high += static_cast<float>(highPoint[1]/(highPoint[1] - lowPoint[1]))*(low - high);
        }
        for (float y : { low, high }) {
            Corner corner;
            wallPoint(*part, angle, y, corner.point);
            if (std::fabs(corner.point[1]) < 1.0) {
                corner.point[1] = 0.0;
            }
            // on the floor but not at an end of the wall: the floor cuts it there
            corner.cut = corner.point[1] < FLOOR_TOLERANCE && std::fabs(y) < part->half[1] - FLOOR_TOLERANCE;
            part->corners.push_back(corner);
        }
    }
}

static std::string format(const char *format, ...) __attribute__((format(printf, 1, 2)));
static std::string format(const char *format, ...)
{
    char buffer[512];
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    return buffer;
}

static std::string pointText(const double *point)
{
    return format("%.2f/%.2f/%.2f", point[0]/1000.0, point[1]/1000.0, point[2]/1000.0);
}

static std::string vectorText(const double *vector)
{
    return format("%.2f/%.2f/%.2f", vector[0], vector[1], vector[2]);
}

static std::string anglesText(const Part &part)
{
    if (part.fullCircle) {
        return "all around";
    }
    return format("%.0f-%.0f degrees", part.angleBegin, std::fmod(part.angleEnd, 360.0));
}

static const char *const CUBE_FACES[6] = { "+x", "-x", "+y", "-y", "+z", "-z" };

std::string summary(const Part &part)
{
    const std::string object = format("%s (surface %u)", shapeName(part.shape), part.surface);
    size_t cuts = 0;
    for (const Corner &corner : part.corners) {
        cuts += corner.cut ? 1 : 0;
    }
    switch (part.type) {
    case TypeFloor:
        return "floor";
    case TypeSideWall:
        return format("side wall %s of the %s: %zu corners%s", CUBE_FACES[part.index%6], object.c_str(),
                      part.corners.size(), cuts > 0 ? format(", %zu cut by the floor", cuts).c_str() : "");
    case TypeCurvedWall:
    case TypeConeWall:
        return format("%s of the %s: radii %.2f/%.2f m, height %.2f m, visible %s%s", typeName(part.type),
                      object.c_str(), part.half[0]/1000.0, part.half[2]/1000.0, 2.0*part.half[1]/1000.0,
                      anglesText(part).c_str(), part.floorCut ? ", cut by the floor" : "");
    case TypeRoundWall:
        return format("round wall (%s) of the %s: radii %.2f/%.2f m%s", part.shape == ShapeCylinder && part.index == 1
                      ? "top" : (part.shape == ShapeCone ? "base" : "bottom"), object.c_str(), part.half[0]/1000.0,
                      part.half[2]/1000.0, part.floorCut ? ", cut by the floor" : "");
    case TypeBall:
        return format("ball (surface %u): radii %.2f/%.2f/%.2f m, visible %s%s", part.surface, part.half[0]/1000.0,
                      part.half[1]/1000.0, part.half[2]/1000.0, anglesText(part).c_str(),
                      part.floorCut ? ", cut by the floor" : "");
    case TypeFreeForm:
        return format("free-form %s (surface %u)", shapeName(part.shape), part.surface);
    default:
        return "none";
    }
}

std::string geometry(const Part &part)
{
    std::string text;
    switch (part.type) {
    case TypeFreeForm:
        return "box center " + pointText(part.objectCenter) + " m, size " + format("%.2f/%.2f/%.2f m",
               part.half[0]/500.0, part.half[1]/500.0, part.half[2]/500.0);
    case TypeSideWall:
        text = "center " + pointText(part.center) + " m, normal " + vectorText(part.normal);
        break;
    case TypeRoundWall:
        text = "center " + pointText(part.center) + " m, normal " + vectorText(part.normal) + ", radius directions "
               + vectorText(part.axes[0]) + " and " + vectorText(part.axes[2]) + ", visible " + anglesText(part);
        break;
    case TypeCurvedWall:
    case TypeConeWall:
        text = std::string(part.type == TypeConeWall ? "base center " : "center ") + pointText(part.center) + " m, axis "
               + vectorText(part.normal) + ", radius directions " + vectorText(part.axes[0]) + " and "
               + vectorText(part.axes[2]) + format(", visible heights %.2f-%.2f m of the axis", part.lowY/1000.0,
                                                   part.highY/1000.0);
        break;
    case TypeBall:
        text = "center " + pointText(part.center) + " m, axes " + vectorText(part.axes[0]) + ", "
               + vectorText(part.axes[1]) + ", " + vectorText(part.axes[2])
               + format(", visible heights %.2f-%.2f m of its y axis", part.lowY/1000.0, part.highY/1000.0);
        break;
    default:
        return "";
    }
    if (!part.corners.empty()) {
        const bool wall = part.type == TypeCurvedWall || part.type == TypeConeWall;
        text += wall ? "; begins at " : (part.type == TypeRoundWall ? "; floor cut at " : "; corners ");
        for (size_t i=0;i<part.corners.size();i++) {
            if (wall && i == 2) {
                text += ", ends at ";
            } else if (i > 0) {
                text += wall ? " - " : ", ";
            }
            text += pointText(part.corners[i].point) + (part.corners[i].cut && part.type != TypeRoundWall ? " (cut)" : "");
        }
        text += " m";
    }
    return text;
}

std::string describe(const Part &part)
{
    std::string text = format("id %u, ", part.id) + summary(part);
    const double residual = part.pixels > 0 ? part.residualSum/static_cast<double>(part.pixels) : 0.0;
    text += format("; %zu pixels, distance from it %.1f mm (max %.1f)", part.pixels, residual, part.residualMax);
    const std::string values = geometry(part);
    return values.empty() ? text : text + "; " + values;
}

}
