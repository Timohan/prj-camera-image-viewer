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

// attribute locations of both programs
#define VERTEX_ATTRIBUTE                0
#define TEXTURE_COORDINATE_ATTRIBUTE    1

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

// rgb: pixel index + 1 of the image (texture coordinates are the pixel centers, so floor() gives the nearest pixel),
// alpha: index + 1 of the camera image, 0 (cleared color) is no image
const char *ViewerShader::m_pickFragShader = "uniform vec2 imageSize;" \
        "uniform float imageIndex;" \
        "varying vec2 texCoord;" \
        "void main()" \
        "{" \
        "    float x = min(floor(texCoord.x*imageSize.x), imageSize.x - 1.0);" \
        "    float y = min(floor(texCoord.y*imageSize.y), imageSize.y - 1.0);" \
        "    float pixel = y*imageSize.x + x + 1.0;" \
        "    float b = floor(pixel/65536.0);" \
        "    float g = floor((pixel - b*65536.0)/256.0);" \
        "    float r = pixel - b*65536.0 - g*256.0;" \
        "    gl_FragColor = vec4(r, g, b, imageIndex + 1.0)/255.0;" \
        "}";

ViewerShader::ViewerShader() :
    m_program(0),
    m_pickProgram(0),
    m_vertexAttribute(-1),
    m_textureCoordinateAttribute(-1)
{
}

ViewerShader::~ViewerShader()
{
    if (m_program) {
        glDeleteProgram(m_program);
    }
    if (m_pickProgram) {
        glDeleteProgram(m_pickProgram);
    }
}

/*!
 * \brief ViewerShader::init
 * compiles and links the shaders
 * \return 0 on success
 */
int ViewerShader::init()
{
    if (linkProgram(m_fragShader, &m_program) != 0 || linkProgram(m_pickFragShader, &m_pickProgram) != 0) {
        return -1;
    }
    m_vertexAttribute = VERTEX_ATTRIBUTE;
    m_textureCoordinateAttribute = TEXTURE_COORDINATE_ATTRIBUTE;
    return 0;
}

/*!
 * \brief ViewerShader::linkProgram
 * links the vertex shader and the fragment shader to a program, the attributes get the same locations in every program
 * \return 0 on success
 */
int ViewerShader::linkProgram(const char *fragSource, GLuint *program)
{
    GLuint vertShader, fragShader;
    if (compileShader(m_vertShader, GL_VERTEX_SHADER, &vertShader) != 0) {
        return -1;
    }
    if (compileShader(fragSource, GL_FRAGMENT_SHADER, &fragShader) != 0) {
        glDeleteShader(vertShader);
        return -1;
    }
    const GLuint handle = glCreateProgram();
    glAttachShader(handle, vertShader);
    glAttachShader(handle, fragShader);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    glBindAttribLocation(handle, VERTEX_ATTRIBUTE, "vertex");
    glBindAttribLocation(handle, TEXTURE_COORDINATE_ATTRIBUTE, "tex_coord");
    glLinkProgram(handle);
    GLint status;
    glGetProgramiv(handle, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        TG_ERROR_LOG("Linking the shader failed");
        glDeleteProgram(handle);
        return -1;
    }
    *program = handle;
    return 0;
}

GLuint ViewerShader::program() const
{
    return m_program;
}

GLuint ViewerShader::pickProgram() const
{
    return m_pickProgram;
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
