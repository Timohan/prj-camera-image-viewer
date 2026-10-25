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
# the vision transformer of the learner is compiled from its sources, so the viewer reads the same model files
LEARNER_PATH?=$(CURRENT_DIR)../prj-camera-image-learner
PKGFLAGS=`pkg-config --cflags --libs prj-tg-ui-lib opencv4`
CXXFLAGS+=-g -O2 -Wall -pedantic -c -pipe -std=gnu++17 -W -D_REENTRANT -fPIC
CXXFLAGS+=-I./src -I$(LEARNER_PATH)/src
CXXFLAGS+=$(PKGFLAGS)
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

learner_OBJS:=$(CURRENT_DIR)src/learner/vision_transformer.o

all: default

default: $(src_OBJS) $(math_OBJS) $(shader_OBJS) $(scene_OBJS) $(learner_OBJS)
	$(CXX) $(src_OBJS) $(math_OBJS) $(shader_OBJS) $(scene_OBJS) $(learner_OBJS) $(LDFLAGS) -o $(TARGET)

$(src_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(math_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(shader_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(scene_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(learner_OBJS): $(LEARNER_PATH)/src/network/vision_transformer.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
	rm -f $(src_SRCDIR)/*.o
	rm -f $(math_SRCDIR)/*.o
	rm -f $(shader_SRCDIR)/*.o
	rm -f $(scene_SRCDIR)/*.o
	rm -rf $(CURRENT_DIR)src/learner
