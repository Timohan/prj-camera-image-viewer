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
 * textured triangles without lights, the camera images already have their lights
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
    GLint vertexAttribute() const;
    GLint textureCoordinateAttribute() const;

private:
    static const char *m_vertShader;
    static const char *m_fragShader;
    GLuint m_program;
    GLint m_vertexAttribute;
    GLint m_textureCoordinateAttribute;

    static int compileShader(const char *source, GLenum type, GLuint *shader);
};

#endif // VIEWER_SHADER_H
