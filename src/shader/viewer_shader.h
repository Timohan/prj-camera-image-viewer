/*!
 * \file
 * \brief file viewer_shader.h
 *
 * Compile shader
 *
 * Copyright of Timo Hannukkala. All rights reserved.
 *
 * \author Timo Hannukkala <timohannukkala@hotmail.com>
 */

#ifndef VIEWER_SHADER_H
#define VIEWER_SHADER_H

#include <GL/glew.h>
#include <GL/gl.h>

/*!
 * \brief ViewerShader
 * textured triangles without lights, the camera images already have their lights.
 * The pick program draws the same triangles with the camera and the image pixel coded in the color
 * (see ViewerScene::pick()), both programs have the same attribute locations, so the meshes work with both.
 */
class ViewerShader
{
public:
    ViewerShader();
    ~ViewerShader();
    ViewerShader(const ViewerShader &) = delete;
    ViewerShader &operator=(const ViewerShader &) = delete;

    int init();
    GLuint program() const;
    GLuint pickProgram() const;
    GLint vertexAttribute() const;
    GLint textureCoordinateAttribute() const;

private:
    static const char *m_vertShader;
    static const char *m_fragShader;
    static const char *m_pickFragShader;
    GLuint m_program;
    GLuint m_pickProgram;
    GLint m_vertexAttribute;
    GLint m_textureCoordinateAttribute;

    static int compileShader(const char *source, GLenum type, GLuint *shader);
    static int linkProgram(const char *fragSource, GLuint *program);
};

#endif // VIEWER_SHADER_H
