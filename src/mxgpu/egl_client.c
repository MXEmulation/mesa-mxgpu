/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#define _POSIX_C_SOURCE 200809L
#include "scene_common.h"
#include "egl_scene.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

static GLuint compile(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    if (!shader)
        return 0;
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "%s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint program(const char *fragment)
{
    const char *vertex =
        "attribute vec4 position_uv; varying vec2 uv;"
        "void main(){gl_Position=vec4(position_uv.xy,0.0,1.0);uv=position_uv.zw;}";
    GLuint vs = compile(GL_VERTEX_SHADER, vertex), fs = compile(GL_FRAGMENT_SHADER, fragment);
    GLuint result = 0;
    if (vs && fs) {
        result = glCreateProgram();
        glAttachShader(result, vs);
        glAttachShader(result, fs);
        glBindAttribLocation(result, 0, "position_uv");
        glLinkProgram(result);
        GLint status;
        glGetProgramiv(result, GL_LINK_STATUS, &status);
        if (!status) {
            char log[2048];
            glGetProgramInfoLog(result, sizeof(log), NULL, log);
            fprintf(stderr, "%s\n", log);
            glDeleteProgram(result);
            result = 0;
        }
    }
    if (vs) glDeleteShader(vs);
    if (fs) glDeleteShader(fs);
    return result;
}

static int frame(unsigned char *pixels)
{
    const char *counter = getenv("MXGPU_SUBMIT_FILE");
    if (counter && (unlink(counter) && errno != ENOENT))
        return 0;
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glReadPixels(0, 0, SCENE_W, SCENE_H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "GL error 0x%x\n", error);
        return 0;
    }
    if (counter) {
        FILE *input = fopen(counter, "r");
        unsigned submits = 0;
        if (!input)
            return 0;
        int valid = fscanf(input, "%u", &submits) == 1 && submits == 1;
        fclose(input);
        if (!valid)
            return 0;
    }
    return 1;
}

static int pixels_match(const unsigned char *pixels, int constant_uv)
{
    for (unsigned y = 0; y < SCENE_H; y++) {
        for (unsigned x = 0; x < SCENE_W; x++) {
            unsigned texel = constant_uv ? 0 : (y >= SCENE_H / 2u ? 2u : 0u) +
                                              (x >= SCENE_W / 2u ? 1u : 0u);
            const unsigned char *actual = pixels + (y * SCENE_W + x) * 4u;
            if (memcmp(actual, scene_texture + texel * 4u, 4)) {
                fprintf(stderr, "Pixel %u,%u differs\n", x, y);
                return 0;
            }
        }
    }
    return 1;
}

int mx_gl_scene_run(int stress)
{
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    GLuint programs[2] = {0}, texture = 0, buffer = 0;
    int current = 0, initialized = 0, status = 1;
    unsigned char pixels[SCENE_W * SCENE_H * 4];
    PFNEGLGETPLATFORMDISPLAYEXTPROC platform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!platform)
        goto cleanup;
    display = platform(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL))
        goto cleanup;
    initialized = 1;
    if (!eglBindAPI(EGL_OPENGL_ES_API))
        goto cleanup;
    EGLConfig configuration;
    EGLint count;
    const EGLint attributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE
    };
    if (!eglChooseConfig(display, attributes, &configuration, 1, &count) || !count)
        goto cleanup;
    context = eglCreateContext(display, configuration, EGL_NO_CONTEXT,
                               (const EGLint[]){EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE});
    surface = eglCreatePbufferSurface(display, configuration,
                                     (const EGLint[]){EGL_WIDTH, SCENE_W, EGL_HEIGHT, SCENE_H, EGL_NONE});
    if (context == EGL_NO_CONTEXT || surface == EGL_NO_SURFACE ||
        !eglMakeCurrent(display, surface, surface, context))
        goto cleanup;
    current = 1;
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    const char *vendor = (const char *)glGetString(GL_VENDOR);
    if (renderer && vendor)
        printf("vendor %s renderer %s\n", vendor, renderer);
    if (!renderer || !vendor || strcmp(vendor, "MX") || strcmp(renderer, "mxgpu"))
        goto cleanup;
    programs[0] = program("precision mediump float; varying vec2 uv; uniform sampler2D source_texture;"
                           "void main(){gl_FragColor=texture2D(source_texture,uv);}");
    programs[1] = program("precision mediump float; uniform sampler2D source_texture;"
                           "void main(){gl_FragColor=texture2D(source_texture,vec2(0.0));}");
    if (!programs[0] || !programs[1])
        goto cleanup;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, scene_texture);
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(scene_vertices), scene_vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
    glViewport(0, 0, SCENE_W, SCENE_H);
    glUseProgram(programs[0]);
    glUniform1i(glGetUniformLocation(programs[0], "source_texture"), 0);
    unsigned first = 0;
    unsigned frames = stress ? 121 : 1;
    for (unsigned i = 0; i < frames; i++) {
        struct timespec before, after;
        if (clock_gettime(CLOCK_MONOTONIC, &before) || !frame(pixels) ||
            !pixels_match(pixels, 0) || clock_gettime(CLOCK_MONOTONIC, &after))
            goto cleanup;
        unsigned hash = scene_hash(pixels, sizeof(pixels));
        if (i == 0)
            first = hash;
        double elapsed = scene_ms(&before, &after);
        printf("frame %u hash %08x completion_ms %.3f\n", i + 1u, hash, elapsed);
        if (hash != first || (stress && i && elapsed >= 100.0))
            goto cleanup;
    }
    if (stress) {
        glDrawArrays(0xffffffffu, 0, 6);
        if (glGetError() != GL_INVALID_ENUM || !frame(pixels) || !pixels_match(pixels, 0))
            goto cleanup;
        puts("PASS: valid draw after rejected primitive mode");
    } else {
        scene_print_pixels(pixels);
        glUseProgram(programs[1]);
        glUniform1i(glGetUniformLocation(programs[1], "source_texture"), 0);
        if (!frame(pixels) || !pixels_match(pixels, 1))
            goto cleanup;
        puts("PASS: texture coordinates and constant texture coordinates");
    }
    status = 0;
cleanup:
    if (status)
        fprintf(stderr, "EGL scene failed, EGL error 0x%x\n", eglGetError());
    if (current) {
        if (buffer) glDeleteBuffers(1, &buffer);
        if (texture) glDeleteTextures(1, &texture);
        if (programs[0]) glDeleteProgram(programs[0]);
        if (programs[1]) glDeleteProgram(programs[1]);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
    if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
    if (initialized) eglTerminate(display);
    return status;
}

#ifdef MXGPU_EGL_CLIENT_MAIN
int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "stress")))
        return 2;
    return mx_gl_scene_run(argc == 2);
}
#endif
