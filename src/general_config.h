#ifndef GENERAL_CONFIG_H
#define GENERAL_CONFIG_H

#include <string>

/*!
 * \brief GeneralConfig
 * Reads configs/general.ini of the viewer, lines are key=value, lines starting with # are comments.
 * cameraImageOutputDirectory is the directory of the camera images, objects and models (same as in
 * prj-camera-image-creator), the camera images are in <cameraImageOutputDirectory>/camera_output and the models are in
 * <cameraImageOutputDirectory>/camera_model.
 */
class GeneralConfig
{
public:
    // returns false and prints the error if the file cannot be read or cameraImageOutputDirectory is missing
    bool read(const std::string &filename);

    const std::string &getCameraImageOutputDirectory() const;
    // <cameraImageOutputDirectory>/camera_output, empty if cameraImageOutputDirectory is missing
    std::string getImageDirectory() const;
    // <cameraImageOutputDirectory>/camera_model, empty if cameraImageOutputDirectory is missing
    std::string getModelDirectory() const;

private:
    std::string getSubdirectory(const char *subdirectory) const;

    std::string m_cameraImageOutputDirectory;
};

#endif // GENERAL_CONFIG_H
