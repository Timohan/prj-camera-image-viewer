#include "general_config.h"
#include <filesystem>
#include <fstream>
#include "global_log.h"

#define CAMERA_IMAGE_OUTPUT_DIRECTORY_KEY   "cameraImageOutputDirectory"
// subdirectories of cameraImageOutputDirectory: camera images of prj-camera-image-creator and models
#define IMAGE_SUBDIRECTORY                  "camera_output"
#define MODEL_SUBDIRECTORY                  "camera_model"

static std::string trim(const std::string &text)
{
    const char *spaces = " \t\r\n";
    const size_t begin = text.find_first_not_of(spaces);
    if (begin == std::string::npos) {
        return "";
    }
    return text.substr(begin, text.find_last_not_of(spaces) - begin + 1);
}

/*!
 * \brief GeneralConfig::read
 * \param filename full path of general.ini
 * \return true if cameraImageOutputDirectory was read
 */
bool GeneralConfig::read(const std::string &filename)
{
    m_cameraImageOutputDirectory.clear();
    std::ifstream file(filename);
    if (!file.is_open()) {
        TG_ERROR_LOG("Cannot open", filename);
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t equal = line.find('=');
        if (equal != std::string::npos && trim(line.substr(0, equal)) == CAMERA_IMAGE_OUTPUT_DIRECTORY_KEY) {
            m_cameraImageOutputDirectory = trim(line.substr(equal + 1));
        }
    }
    if (m_cameraImageOutputDirectory.empty()) {
        TG_ERROR_LOG(filename + ":", CAMERA_IMAGE_OUTPUT_DIRECTORY_KEY, "is missing");
        return false;
    }
    return true;
}

const std::string &GeneralConfig::getCameraImageOutputDirectory() const
{
    return m_cameraImageOutputDirectory;
}

std::string GeneralConfig::getImageDirectory() const
{
    return getSubdirectory(IMAGE_SUBDIRECTORY);
}

std::string GeneralConfig::getModelDirectory() const
{
    return getSubdirectory(MODEL_SUBDIRECTORY);
}

/*!
 * \brief GeneralConfig::getSubdirectory
 * \return <cameraImageOutputDirectory>/subdirectory, empty if cameraImageOutputDirectory is missing
 */
std::string GeneralConfig::getSubdirectory(const char *subdirectory) const
{
    if (m_cameraImageOutputDirectory.empty()) {
        return std::string();
    }
    return (std::filesystem::path(m_cameraImageOutputDirectory) / subdirectory).string();
}
