#pragma once

#include <glad/glad.h>

class CPUProcessor
{
  public:
    CPUProcessor(int w, int h);
    ~CPUProcessor();

  public:
    void
    run();

  private:
    int _width;
    int _height;

    unsigned char *_pixels;

    GLuint _pbo[2];
    GLuint _shader;

    GLuint _quadVbo;
    GLuint _quadEbo;
    GLuint _quadVao;

};
