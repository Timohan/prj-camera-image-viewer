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
#include <string>
#include "mainwindow.h"

#define IMAGE_DIRECTORY_ARGUMENT    "--imageDirectory="
#define SCENE_ARGUMENT              "--scene="
#define MODEL_FILE_ARGUMENT         "--modelFile="
// output directory of prj-camera-image-creator
#define DEFAULT_IMAGE_DIRECTORY     "/home/timo/camera_output"
// trained model of prj-camera-image-learner
#define DEFAULT_MODEL_FILE          "/home/timo/camera_model/vision_transformer.model"

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
 * \brief main
 * \param argc
 * \param argv --imageDirectory=<directory> images of prj-camera-image-creator, default is DEFAULT_IMAGE_DIRECTORY,
 * --scene=<name> images <name>_<camera number>.png and .dat, for example objects999_20261025-175558,
 * default is the scene that has the largest number,
 * --modelFile=<file> model of prj-camera-image-learner, default is DEFAULT_MODEL_FILE
 * \return 0 on success
 */
int main(int argc , char *argv[])
{
    std::string imageDirectory = DEFAULT_IMAGE_DIRECTORY;
    std::string scene;
    std::string modelFile = DEFAULT_MODEL_FILE;
    for (int i=1;i<argc;i++) {
        if (!getArgument(argv[i], IMAGE_DIRECTORY_ARGUMENT, &imageDirectory) && !getArgument(argv[i], SCENE_ARGUMENT, &scene)
            && !getArgument(argv[i], MODEL_FILE_ARGUMENT, &modelFile)) {
            fprintf(stderr, "Unknown argument %s, usage: prj-camera-image-viewer [--imageDirectory=<directory>] "
                            "[--scene=<name>] [--modelFile=<file>]\n", argv[i]);
            return 1;
        }
    }
    static MainWindow m_mainwindow(1920, 1080);
    if (!m_mainwindow.load(imageDirectory, scene, modelFile)) {
        return 1;
    }
    return m_application.exec();
}
