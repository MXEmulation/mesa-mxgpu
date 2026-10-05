/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int extension(const char *list, const char *name)
{
    size_t length = strlen(name);
    const char *at = list;
    while (at && (at = strstr(at, name))) {
        if ((at == list || at[-1] == ' ') && (!at[length] || at[length] == ' '))
            return 1;
        at += length;
    }
    return 0;
}

static GLuint shader(GLenum type, const char *source)
{
    GLuint object = glCreateShader(type);
    if (!object)
        return 0;
    glShaderSource(object, 1, &source, NULL);
    glCompileShader(object);
    GLint valid;
    glGetShaderiv(object, GL_COMPILE_STATUS, &valid);
    if (!valid) {
        char message[2048];
        glGetShaderInfoLog(object, sizeof(message), NULL, message);
        fprintf(stderr, "%s\n", message);
        glDeleteShader(object);
        return 0;
    }
    return object;
}

static int error_is(const char *name, GLenum expected)
{
    GLenum actual = glGetError();
    if (actual != expected) {
        fprintf(stderr, "%s error 0x%x expected 0x%x\n", name, actual, expected);
        return 0;
    }
    return 1;
}

static int pixels_match(const char *name, int phase)
{
    unsigned char pixels[32 * 16 * 4];
    glReadPixels(0, 0, 32, 16, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (!error_is(name, GL_NO_ERROR))
        return 0;
    const unsigned char red[] = {255, 0, 0, 255}, green[] = {0, 255, 0, 255};
    const unsigned char white[] = {255, 255, 255, 255};
    const unsigned char black[] = {0, 0, 0, 255}, blue[] = {0, 0, 255, 255};
    const unsigned char clear[] = {64, 128, 191, 255};
    for (unsigned y = 0; y < 16; y++) {
        for (unsigned x = 0; x < 32; x++) {
            const unsigned char *expected = phase == 2 ? clear : phase == 3 ? blue : black;
            if (phase < 2 && y > 0 && y < 15) {
                if (x > 0 && x < 15)
                    expected = phase == 1 ? green : red;
                if (x > 16 && x < 31)
                    expected = phase == 1 ? white : green;
            }
            if (memcmp(pixels + (y * 32 + x) * 4, expected, 4)) {
                fprintf(stderr, "%s differs at %u,%u\n", name, x, y);
                return 0;
            }
        }
    }
    printf("PASS %s: 512 pixels\n", name);
    return 1;
}

int main(void)
{
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    GLuint shaders[2] = {0}, program = 0, buffers[4] = {0};
    int initialized = 0, current = 0, status = 1;
#define REQUIRE(call) do { if (!(call)) { fprintf(stderr, "Failed %s\n", #call); goto cleanup; } } while (0)
    PFNEGLGETPLATFORMDISPLAYEXTPROC platform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    REQUIRE(platform);
    display = platform(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    REQUIRE(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL));
    initialized = 1;
    REQUIRE(eglBindAPI(EGL_OPENGL_ES_API));
    EGLConfig configuration;
    EGLint configurations;
    REQUIRE(eglChooseConfig(display, (const EGLint[]){
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE
    }, &configuration, 1, &configurations) && configurations);
    context = eglCreateContext(display, configuration, EGL_NO_CONTEXT,
                               (const EGLint[]){EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE});
    surface = eglCreatePbufferSurface(display, configuration,
                                     (const EGLint[]){EGL_WIDTH, 32, EGL_HEIGHT, 16, EGL_NONE});
    REQUIRE(context != EGL_NO_CONTEXT && surface != EGL_NO_SURFACE &&
            eglMakeCurrent(display, surface, surface, context));
    current = 1;
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    const char *vendor = (const char *)glGetString(GL_VENDOR);
    const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
    REQUIRE(renderer && vendor && !strcmp(renderer, "mxgpu") && !strcmp(vendor, "MX"));
    printf("vendor %s renderer %s version %s\n", vendor, renderer, glGetString(GL_VERSION));
    REQUIRE(extension(extensions, "GL_ANGLE_instanced_arrays"));
    PFNGLDRAWARRAYSINSTANCEDANGLEPROC arrays =
        (PFNGLDRAWARRAYSINSTANCEDANGLEPROC)eglGetProcAddress("glDrawArraysInstancedANGLE");
    PFNGLDRAWELEMENTSINSTANCEDANGLEPROC elements =
        (PFNGLDRAWELEMENTSINSTANCEDANGLEPROC)eglGetProcAddress("glDrawElementsInstancedANGLE");
    PFNGLVERTEXATTRIBDIVISORANGLEPROC divisor =
        (PFNGLVERTEXATTRIBDIVISORANGLEPROC)eglGetProcAddress("glVertexAttribDivisorANGLE");
    PFNGLDRAWARRAYSINSTANCEDEXTPROC ext_arrays =
        (PFNGLDRAWARRAYSINSTANCEDEXTPROC)eglGetProcAddress("glDrawArraysInstancedEXT");
    REQUIRE(arrays && elements && divisor && ext_arrays &&
            extension(extensions, "GL_EXT_draw_instanced"));
    shaders[0] = shader(GL_VERTEX_SHADER,
        "attribute vec2 position;attribute vec2 offset;attribute vec4 color;varying vec4 paint;"
        "void main(){paint=color;gl_Position=vec4(position+offset,0.0,1.0);}");
    shaders[1] = shader(GL_FRAGMENT_SHADER,
        "precision mediump float;varying vec4 paint;void main(){gl_FragColor=paint;}");
    REQUIRE(shaders[0] && shaders[1]);
    program = glCreateProgram();
    REQUIRE(program);
    glAttachShader(program, shaders[0]);
    glAttachShader(program, shaders[1]);
    glBindAttribLocation(program, 0, "position");
    glBindAttribLocation(program, 1, "offset");
    glBindAttribLocation(program, 2, "color");
    glLinkProgram(program);
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char message[2048];
        glGetProgramInfoLog(program, sizeof(message), NULL, message);
        fprintf(stderr, "%s\n", message);
        goto cleanup;
    }
    glUseProgram(program);
    const float vertices[] = {999, 999, -.45f, -.9f, .45f, -.9f, .45f, .9f,
                              -.45f, -.9f, .45f, .9f, -.45f, .9f};
    const float offsets[] = {999, 999, -.5f, 0, .5f, 0};
    const float colors[] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1};
    const void *data[] = {vertices, offsets, colors};
    const GLsizeiptr sizes[] = {sizeof(vertices), sizeof(offsets), sizeof(colors)};
    glGenBuffers(4, buffers);
    for (unsigned i = 0; i < 3; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, buffers[i]);
        glBufferData(GL_ARRAY_BUFFER, sizes[i], data[i], GL_STATIC_DRAW);
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, i == 2 ? 4 : 2, GL_FLOAT, GL_FALSE, 0,
                              (const void *)(uintptr_t)(i == 1 ? 2 * sizeof(float) : 0));
        divisor(i, i == 0 ? 0 : 1);
    }
    GLint queried;
    GLfloat queried_float;
    divisor(1, 7);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE, &queried);
    glGetVertexAttribfv(1, GL_VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE, &queried_float);
    REQUIRE(queried == 7 && queried_float == 7 && error_is("divisor query", GL_NO_ERROR));
    divisor(1, 1);
    glViewport(0, 0, 32, 16);
    glDisable(GL_DITHER);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    arrays(GL_TRIANGLES, 1, 6, 2);
    REQUIRE(pixels_match("arrays first=1 divisor=1", 0));
    const uint8_t u8[] = {99, 1, 2, 3, 4, 5, 6};
    const uint16_t u16[] = {99, 1, 2, 3, 4, 5, 6};
    const uint32_t u32[] = {99, 1, 2, 3, 4, 5, 6};
    const void *indices[] = {u8, u16, u32};
    const GLenum types[] = {GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT};
    const unsigned bytes[] = {1, 2, 4};
    unsigned variants = extension(extensions, "GL_OES_element_index_uint") ? 3 : 2;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[3]);
    for (unsigned i = 0; i < variants; i++) {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 7 * bytes[i], indices[i], GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        elements(GL_TRIANGLES, 6, types[i], (const void *)(uintptr_t)bytes[i], 2);
        REQUIRE(pixels_match(i == 0 ? "indices u8 offset" : i == 1 ?
                             "indices u16 offset" : "indices u32 offset", 0));
    }
    divisor(1, 2);
    glClear(GL_COLOR_BUFFER_BIT);
    arrays(GL_TRIANGLES, 1, 6, 4);
    REQUIRE(pixels_match("divisor=2 advances every second instance", 1));
    glClearColor(.25f, .5f, .75f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    arrays(GL_TRIANGLES, 1, 6, 0);
    REQUIRE(pixels_match("zero instances", 2));
    GLint maximum;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maximum);
    divisor((GLuint)maximum, 1);
    REQUIRE(error_is("divisor index limit", GL_INVALID_VALUE));
    arrays(GL_TRIANGLES, -1, 6, 1);
    REQUIRE(error_is("negative first", GL_INVALID_VALUE));
    arrays(GL_TRIANGLES, 1, -1, 1);
    REQUIRE(error_is("negative count", GL_INVALID_VALUE));
    arrays(GL_TRIANGLES, 1, 6, -1);
    REQUIRE(error_is("negative instances", GL_INVALID_VALUE));
    arrays(0xffffffffu, 1, 6, 1);
    REQUIRE(error_is("invalid mode", GL_INVALID_ENUM));
    elements(GL_TRIANGLES, 6, GL_FLOAT, 0, 1);
    REQUIRE(error_is("invalid index type", GL_INVALID_ENUM));
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    divisor(0, 1);
    arrays(GL_TRIANGLES, 1, 6, 2);
    REQUIRE(error_is("all active arrays instanced", GL_INVALID_OPERATION));
    elements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0, 2);
    REQUIRE(error_is("indexed all active arrays instanced", GL_INVALID_OPERATION));
    arrays(GL_TRIANGLES, 1, 0, 0);
    REQUIRE(error_is("empty draw still needs divisor zero", GL_INVALID_OPERATION));
    REQUIRE(maximum > 3);
    GLuint inactive = (GLuint)maximum - 1;
    glEnableVertexAttribArray(inactive);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glVertexAttribPointer(inactive, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    divisor(inactive, 0);
    arrays(GL_TRIANGLES, 1, 6, 2);
    REQUIRE(error_is("inactive divisor zero", GL_INVALID_OPERATION));
    glDisableVertexAttribArray(inactive);
    divisor(0, 0);
    glDisableVertexAttribArray(0);
    arrays(GL_TRIANGLES, 1, 6, 2);
    REQUIRE(error_is("disabled divisor zero", GL_INVALID_OPERATION));
    ext_arrays(GL_TRIANGLES, 1, 6, 2);
    REQUIRE(error_is("EXT draw retains its own semantics", GL_NO_ERROR));
    REQUIRE(pixels_match("rejected ANGLE draws preserve pixels", 3));
    glEnableVertexAttribArray(0);
    divisor(1, 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    arrays(GL_TRIANGLES, 1, 6, 2);
    REQUIRE(pixels_match("valid draw after ANGLE errors", 0));
    status = 0;
cleanup:
    if (current) {
        glDeleteBuffers(4, buffers);
        if (program) glDeleteProgram(program);
        if (shaders[0]) glDeleteShader(shaders[0]);
        if (shaders[1]) glDeleteShader(shaders[1]);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
    if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
    if (initialized) eglTerminate(display);
#undef REQUIRE
    return status;
}
