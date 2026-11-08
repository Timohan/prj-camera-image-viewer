#/*!
#* \file Makefile
#* \brief Makefile for compiling
#*
#* Copyright of Timo hannukkala, Inc. All rights reserved.
#*
#* \author Timo Hannukkala <timohannukkala@hotmail.com>
#*/
TARGET:=prj-camera-image-viewer
CXX:=$(if $(CXX),$(CXX),g++)
# src/network and src/global_log.h are copies of the sources of prj-camera-image-learner, so the viewer reads the same
# model files and it can be compiled without the learner, `make update-learner` copies them again from LEARNER_PATH
LEARNER_PATH?=$(CURRENT_DIR)../prj-camera-image-learner
LEARNER_FILES:=network/vision_transformer.h network/vision_transformer.cpp network/pixel_label.h global_log.h \
               stereo/stereo_matcher.h stereo/stereo_matcher.cpp
PKGFLAGS=`pkg-config --cflags --libs prj-tg-ui-lib opencv4`
CXXFLAGS+=-g -O2 -Wall -pedantic -c -pipe -std=gnu++17 -W -D_REENTRANT -fPIC
CXXFLAGS+=-I./src
CXXFLAGS+=$(PKGFLAGS)
# model of this repository, it is used if no model is found from <cameraImageOutputDirectory>/camera_model
CXXFLAGS+=-DDEFAULT_MODEL_FILE=\"$(CURRENT_DIR)model/default.model\"
# configs/general.ini of this repository
CXXFLAGS+=-DCONFIGS_PATH=\"$(CURRENT_DIR)configs/\"
LDFLAGS:=$(PKGFLAGS) -lGL -lGLEW -lglfw -lpthread
# set current make dir
CURRENT_DIR=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

src_SRCDIR:=$(CURRENT_DIR)src
src_SRCS:=$(wildcard $(src_SRCDIR)/*.cpp)
src_OBJS:=$(src_SRCS:.cpp=.o)

math_SRCDIR:=$(CURRENT_DIR)src/math
math_SRCS:=$(wildcard $(math_SRCDIR)/*.cpp)
math_OBJS:=$(math_SRCS:.cpp=.o)

shader_SRCDIR:=$(CURRENT_DIR)src/shader
shader_SRCS:=$(wildcard $(shader_SRCDIR)/*.cpp)
shader_OBJS:=$(shader_SRCS:.cpp=.o)

scene_SRCDIR:=$(CURRENT_DIR)src/scene
scene_SRCS:=$(wildcard $(scene_SRCDIR)/*.cpp)
scene_OBJS:=$(scene_SRCS:.cpp=.o)

network_SRCDIR:=$(CURRENT_DIR)src/network
network_SRCS:=$(wildcard $(network_SRCDIR)/*.cpp)
network_OBJS:=$(network_SRCS:.cpp=.o)

stereo_SRCDIR:=$(CURRENT_DIR)src/stereo
stereo_SRCS:=$(wildcard $(stereo_SRCDIR)/*.cpp)
stereo_OBJS:=$(stereo_SRCS:.cpp=.o)

all: default

default: $(src_OBJS) $(math_OBJS) $(shader_OBJS) $(scene_OBJS) $(network_OBJS) $(stereo_OBJS)
	$(CXX) $(src_OBJS) $(math_OBJS) $(shader_OBJS) $(scene_OBJS) $(network_OBJS) $(stereo_OBJS) $(LDFLAGS) -o $(TARGET)

$(src_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(math_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(shader_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(scene_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(network_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# matching of the cameras like in the learner (-O3)
$(stereo_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -O3 -c $< -o $@

# copies the network sources again from prj-camera-image-learner, run make clean and make after it
update-learner:
	@test -d $(LEARNER_PATH)/src || { echo "Error: $(LEARNER_PATH)/src is missing, give it with make update-learner LEARNER_PATH=<directory>"; exit 1; }
	mkdir -p $(CURRENT_DIR)src/network $(CURRENT_DIR)src/stereo
	for file in $(LEARNER_FILES); do cp $(LEARNER_PATH)/src/$$file $(CURRENT_DIR)src/$$file || exit 1; done

clean:
	rm -f $(TARGET)
	rm -f $(src_SRCDIR)/*.o
	rm -f $(math_SRCDIR)/*.o
	rm -f $(shader_SRCDIR)/*.o
	rm -f $(scene_SRCDIR)/*.o
	rm -f $(network_SRCDIR)/*.o
	rm -f $(stereo_SRCDIR)/*.o
