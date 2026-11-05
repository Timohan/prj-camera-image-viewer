/*!
 * \file
 * \brief file main.cpp
 *
 * 360 degree view of the images of the cameras of prj-camera-image-creator,
 * the distances of the pixels are predicted with the model of prj-camera-image-learner
 *
 * Copyright of Timo Hannukkala. All rights reserved.
 *
 * \author Timo Hannukkala <timo.hannukkala@hotmail.com>
 */

#include <stdio.h>
#include <string.h>
#include <application/tg_application.h>
#include <filesystem>
#include <memory>
#include <string>
#include "mainwindow.h"
#include "general_config.h"

#define IMAGE_DIRECTORY_ARGUMENT    "--imageDirectory="
#define SCENE_ARGUMENT              "--scene="
#define MODEL_FILE_ARGUMENT         "--modelFile="
// images of prj-camera-image-creator are in <cameraImageOutputDirectory of configs/general.ini>/camera_output,
// trained models of prj-camera-image-learner: <model directory>/MODEL_FILE_NAME, then the newest .model file of
// the model directory (<cameraImageOutputDirectory of configs/general.ini>/camera_model), then the model of this
// repository
#ifndef CONFIGS_PATH
#define CONFIGS_PATH                "configs/"
#endif
#define GENERAL_CONFIG_FILE         "general.ini"
#define MODEL_FILE_NAME             "vision_transformer.model"
#define MODEL_FILE_EXTENSION        ".model"
#ifndef DEFAULT_MODEL_FILE
#define DEFAULT_MODEL_FILE          "model/default.model"
#endif

static TgApplication m_application;

TgApplication *getApplication()
{
    return &m_application;
}

/*!
 * \brief getArgument
 * \return true if the argument starts with the name and has a value, the value is returned
 */
static bool getArgument(const char *argument, const char *name, std::string *value)
{
    if (strncmp(argument, name, strlen(name)) != 0 || argument[strlen(name)] == '\0') {
        return false;
    }
    *value = argument + strlen(name);
    return true;
}

/*!
 * \brief findModelFile
 * \param modelDirectory <cameraImageOutputDirectory>/camera_model, empty if it is not known
 * \return modelDirectory/MODEL_FILE_NAME if it exists, otherwise the newest MODEL_FILE_EXTENSION file of
 * modelDirectory, otherwise DEFAULT_MODEL_FILE
 */
static std::string findModelFile(const std::string &modelDirectory)
{
    if (modelDirectory.empty()) {
        return DEFAULT_MODEL_FILE;
    }
    std::error_code error;
    const std::filesystem::path preferred = std::filesystem::path(modelDirectory) / MODEL_FILE_NAME;
    if (std::filesystem::is_regular_file(preferred, error)) {
        return preferred.string();
    }
    std::filesystem::path newest;
    std::filesystem::file_time_type newestTime;
    for (std::filesystem::directory_iterator it(modelDirectory, error), end; !error && it != end; it.increment(error)) {
        if (it->path().extension() != MODEL_FILE_EXTENSION || !it->is_regular_file(error)) {
            continue;
        }
        const std::filesystem::file_time_type time = it->last_write_time(error);
        if (!error && (newest.empty() || time > newestTime)) {
            newest = it->path();
            newestTime = time;
        }
    }
    if (!newest.empty()) {
        return newest.string();
    }
    return DEFAULT_MODEL_FILE;
}

/*!
 * \brief main
 * \param argc
 * \param argv --imageDirectory=<directory> images of prj-camera-image-creator, default is
 * <cameraImageOutputDirectory of configs/general.ini>/camera_output,
 * --scene=<name> images <name>_<camera number>.png and .dat, for example objects999_20261025-175558,
 * default is the scene that has the largest number,
 * --modelFile=<file> model of prj-camera-image-learner, default is from findModelFile()
 * \return 0 on success
 */
int main(int argc , char *argv[])
{
    std::string imageDirectory;
    std::string scene;
    std::string modelFile;
    for (int i=1;i<argc;i++) {
        if (!getArgument(argv[i], IMAGE_DIRECTORY_ARGUMENT, &imageDirectory) && !getArgument(argv[i], SCENE_ARGUMENT, &scene)
            && !getArgument(argv[i], MODEL_FILE_ARGUMENT, &modelFile)) {
            fprintf(stderr, "Unknown argument %s, usage: prj-camera-image-viewer [--imageDirectory=<directory>] "
                            "[--scene=<name>] [--modelFile=<file>]\n", argv[i]);
            return 1;
        }
    }
    if (imageDirectory.empty() || modelFile.empty()) {
        // the directories that are not given are taken from cameraImageOutputDirectory
        GeneralConfig config;
        config.read(std::string(CONFIGS_PATH) + GENERAL_CONFIG_FILE);
        if (imageDirectory.empty()) {
            imageDirectory = config.getImageDirectory();
            if (imageDirectory.empty()) {
                fprintf(stderr, "No image directory, set cameraImageOutputDirectory of %s%s or give --imageDirectory\n",
                        CONFIGS_PATH, GENERAL_CONFIG_FILE);
                return 1;
            }
        }
        if (modelFile.empty()) {
            modelFile = findModelFile(config.getModelDirectory());
            printf("Model file %s\n", modelFile.c_str());
        }
    }
    // images are loaded and their distances predicted before the window is created
    std::unique_ptr<ViewerScene> viewerScene = std::make_unique<ViewerScene>();
    if (!viewerScene->load(imageDirectory, scene, modelFile)) {
        return 1;
    }
    static MainWindow m_mainwindow(1920, 1080, std::move(viewerScene));
    return m_application.exec();
}
