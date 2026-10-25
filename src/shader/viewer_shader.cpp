/*!
 * \file
 * \brief file viewer_shader.cpp
 *
 * Compile shader
 *
 * Copyright of Timo Hannukkala. All rights reserved.
 *
 * \author Timo Hannukkala <timohannukkala@hotmail.com>
 */

#include "viewer_shader.h"
#include "global_log.h"

const char *ViewerShader::m_vertShader = "attribute vec3 vertex;" \
        "attribute vec2 tex_coord;" \
        "varying vec2 texCoord;" \
        "uniform mat4 view;" \
        "uniform mat4 projection;" \
        "void main()" \
        "{" \
        "    texCoord    = tex_coord;" \
        "    gl_Position = projection*(view*vec4(vertex, 1.0));" \
        "}";

const char *ViewerShader::m_fragShader = "uniform sampler2D texture;" \
        "varying vec2 texCoord;" \
        "void main()" \
        "{" \
        "    gl_FragColor = vec4(texture2D(texture, texCoord).rgb, 1.0);" \
        "}";

ViewerShader::ViewerShader() :
    m_program(0),
    m_vertexAttribute(-1),
    m_textureCoordinateAttribute(-1)
{
}

ViewerShader::~ViewerShader()
{
    if (m_program) {
        glDeleteProgram(m_program);
    }
}

/*!
 * \brief ViewerShader::init
 * compiles and links the shaders
 * \return 0 on success
 */
int ViewerShader::init()
{
    GLuint vertShader, fragShader;
    if (compileShader(m_vertShader, GL_VERTEX_SHADER, &vertShader) != 0
        || compileShader(m_fragShader, GL_FRAGMENT_SHADER, &fragShader) != 0) {
        return -1;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        TG_ERROR_LOG("Linking the shader failed");
        glDeleteProgram(program);
        return -1;
    }
    m_program = program;
    m_vertexAttribute = glGetAttribLocation(program, "vertex");
    m_textureCoordinateAttribute = glGetAttribLocation(program, "tex_coord");
    return 0;
}

GLuint ViewerShader::program() const
{
    return m_program;
}

GLint ViewerShader::vertexAttribute() const
{
    return m_vertexAttribute;
}

GLint ViewerShader::textureCoordinateAttribute() const
{
    return m_textureCoordinateAttribute;
}

int ViewerShader::compileShader(const char *source, GLenum type, GLuint *shader)
{
    const GLuint handle = glCreateShader(type);
    if (handle == 0) {
        TG_ERROR_LOG("Creating the shader failed");
        return -1;
    }
    glShaderSource(handle, 1, &source, nullptr);
    glCompileShader(handle);
    GLint status;
    glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        char infoLog[512];
        glGetShaderInfoLog(handle, sizeof(infoLog), nullptr, infoLog);
        TG_ERROR_LOG("Compiling the shader failed:", infoLog);
        glDeleteShader(handle);
        return -1;
    }
    *shader = handle;
    return 0;
}
