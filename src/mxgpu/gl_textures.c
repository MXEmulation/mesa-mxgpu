/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <stdio.h>
#include <stdlib.h>

#define TEXTURES 16

static const char *vertex_source =
    "attribute vec2 position;\n"
    "varying float column;\n"
    "void main()\n"
    "{\n"
    "    column = (position.x + 1.0) * 8.0;\n"
    "    gl_Position = vec4(position, 0.0, 1.0);\n"
    "}\n";

static int setup(void)
{
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    const EGLint config_attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                        EGL_RED_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
    const EGLint surface_attributes[] = {EGL_WIDTH, TEXTURES, EGL_HEIGHT, 1, EGL_NONE};
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLDisplay display;
    EGLConfig config;
    EGLint count = 0;
    if (!get_display)
        return -1;
    display = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL) ||
        !eglChooseConfig(display, config_attributes, &config, 1, &count) || count < 1 || !eglBindAPI(EGL_OPENGL_ES_API))
        return -1;
    EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT)
        return -1;
    return eglMakeCurrent(display, surface, surface, context) ? 0 : -1;
}

static GLuint program(void)
{
    char source[4096];
    int length = snprintf(source, sizeof source, "precision mediump float;\nvarying float column;\n");
    for (int i = 0; i < TEXTURES; i++)
        length += snprintf(source + length, sizeof source - length, "uniform sampler2D t%d;\n", i);
    length += snprintf(source + length, sizeof source - length,
                       "void main()\n{\n    float x = floor(column);\n    vec4 color = vec4(0.0);\n");
    for (int i = 0; i < TEXTURES; i++)
        length += snprintf(source + length, sizeof source - length,
                           "    color += texture2D(t%d, vec2(0.5)) * clamp(1.0 - abs(x - %d.0), 0.0, 1.0);\n", i, i);
    snprintf(source + length, sizeof source - length, "    gl_FragColor = color;\n}\n");
    const char *sources[2] = {vertex_source, source};
    const GLenum kinds[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint handle = glCreateProgram();
    GLint linked = 0;
    for (int i = 0; i < 2; i++) {
        GLuint shader = glCreateShader(kinds[i]);
        glShaderSource(shader, 1, &sources[i], NULL);
        glCompileShader(shader);
        glAttachShader(handle, shader);
    }
    glBindAttribLocation(handle, 0, "position");
    glLinkProgram(handle);
    glGetProgramiv(handle, GL_LINK_STATUS, &linked);
    return linked ? handle : 0;
}

int main(void)
{
    static const float triangle[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    unsigned char texels[TEXTURES][4], pixels[TEXTURES][4];
    GLint units = 0;
    GLuint handle;
    if (setup()) {
        printf("gl_textures FAIL: no surfaceless GLES2 context\n");
        return 1;
    }
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &units);
    if (units < TEXTURES) {
        printf("gl_textures FAIL: GL_MAX_TEXTURE_IMAGE_UNITS is %d\n", units);
        return 1;
    }
    if (!(handle = program())) {
        printf("gl_textures FAIL: the 16-sampler program did not link\n");
        return 1;
    }
    glUseProgram(handle);
    for (int i = 0; i < TEXTURES; i++) {
        GLuint texture;
        char name[8];
        texels[i][0] = (unsigned char)(i * 16 + 8);
        texels[i][1] = (unsigned char)(255 - i * 16);
        texels[i][2] = (unsigned char)(i * 37);
        texels[i][3] = 255;
        glActiveTexture(GL_TEXTURE0 + i);
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        snprintf(name, sizeof name, "t%d", i);
        glUniform1i(glGetUniformLocation(handle, name), i);
    }
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, triangle);
    glEnableVertexAttribArray(0);
    glViewport(0, 0, TEXTURES, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(0, 0, TEXTURES, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    for (int i = 0; i < TEXTURES; i++)
        for (int c = 0; c < 4; c++)
            if (abs(pixels[i][c] - texels[i][c]) > 1) {
                printf("gl_textures FAIL: column %d is %u %u %u %u, expected %u %u %u %u\n", i, pixels[i][0],
                       pixels[i][1], pixels[i][2], pixels[i][3], texels[i][0], texels[i][1], texels[i][2], texels[i][3]);
                return 1;
            }
    printf("gl_textures PASS: %d textures sampled in one draw, GL error %#x\n", TEXTURES, glGetError());
    return 0;
}
