#include <stdexcept>
#include <string>

#include "cpu_processor.h"

namespace
{
  GLuint createQuadShader()
  {
    static const char *v_shader_source =
    {
      "#version 330 core\n"
      "layout (location = 0) in vec2 aPos;"

      "out vec2 vUv;"

      "void main()"
      "{"
          "vUv = (aPos + vec2(1.0)) / 2.0;"
          "gl_Position = vec4(aPos, 0.0, 1.0);"
      "}"
    };

    static const char *f_shader_source =
    {
      "#version 330 core\n"

      "in vec2 vUv;"
      "out vec4 FragColor;"

      "uniform sampler2D uTexture;"

      "void main()"
      "{"
          "FragColor = texture(uTexture, vUv);"
      "}"
    };

    int success = 1;
    char info_log[512];

    GLuint vx_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fx_shader = glCreateShader(GL_FRAGMENT_SHADER);

    // Compiles vertex shader
    glShaderSource(vx_shader, 1, &v_shader_source, NULL);
    glCompileShader(vx_shader);
    glGetShaderiv(vx_shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
      glGetShaderInfoLog(vx_shader, 512, NULL, info_log);
      throw std::runtime_error(std::string("artracer: vertex shader: ") + info_log);
    }

    // Compiles vertex shader
    glShaderSource(fx_shader, 1, &f_shader_source, NULL);
    glCompileShader(fx_shader);
    glGetShaderiv(fx_shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
      glGetShaderInfoLog(fx_shader, 512, NULL, info_log);
      throw std::runtime_error(std::string("artracer: fragment shader: ") + info_log);
    }

    GLuint shader = glCreateProgram();
    glAttachShader(shader, vx_shader);
    glAttachShader(shader, fx_shader);
    glLinkProgram(shader);
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success)
    {
      glGetProgramInfoLog(shader, 512, NULL, info_log);
      throw std::runtime_error(std::string("artracer: shader link: ") + info_log);
    }
    return shader;
  }

  GLuint createQuadVAO(GLuint *vbo, GLuint *ebo)
  {
    // Quad vertices
    static GLfloat vertices[8] =
    {
      -1.0, -1.0, // upper-left
      -1.0, 1.0, // lower-left
      1.0, -1.0, // upper-right
      1.0, 1.0, // lower-right
    };

    static GLuint indices[] = {
      0, 1, 3, // first triangle
      0, 2, 3  // second triangle
    };

    GLuint vao = 0;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, vbo);
    glGenBuffers(1, ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof (GLfloat),
      &vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof (indices),
      indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
      2 * sizeof(GL_FLOAT), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return vao;
  }
}

CPUProcessor::CPUProcessor(int w, int h)
              : _width(w)
              , _height(h)
{

  // Creates shader rendering the texture to a
  // space screen quad
  _shader = createQuadShader();

  // Creates VBO hosting the quad vertices
  _quadVao = createQuadVAO(&_quadVbo, &_quadEbo);

  // Creates texture storing the CPU computed
  // pathtracing image.
  glGenTextures(1, &_screen_tex);
  glBindTexture(GL_TEXTURE_2D, _screen_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
    GL_RGB, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Generate ping-pong VBOs
  glGenBuffers(2, &_pbo[0]);
  for (unsigned i = 0; i < 2; ++i)
  {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo[i]);
    glBufferData(
      GL_PIXEL_UNPACK_BUFFER,
      _width * _height * 3,
      NULL, GL_DYNAMIC_DRAW
    );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }

  // DEBUG
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo[0]);
  GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
  //GLubyte *ptr = new GLubyte[w * h * 3];
  if (ptr)
  {
    //glBindTexture(GL_TEXTURE_2D, _screen_tex);
    // update data directly on the mapped buffer
    for (size_t y = 0; y < h; ++y)
    {
      size_t i = y * w * 3;
      if (y % 2 == 0) {
        for (size_t x = 0; x < w * 3; x += 3) {
          ptr[i + x] = 255;
          ptr[i + x + 1] = 0;
          ptr[i + x + 2] = 0;
        }
      }
      else {
        for (size_t x = 0; x < w * 3; x += 3) {
          ptr[i + x] = 0;
          ptr[i + x + 1] = 0;
          ptr[i + x + 2] = 128;
        }
      }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, ptr);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
  }
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glBindTexture(GL_TEXTURE_2D, _screen_tex);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo[0]);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
    GL_RGB, GL_UNSIGNED_BYTE, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  // END DEBUG

}

CPUProcessor::~CPUProcessor()
{
  glDeleteBuffers(2, &_pbo[0]);
  glDeleteBuffers(1, &_quadVao);
  glDeleteBuffers(1, &_quadVbo);
  glDeleteBuffers(1, &_quadEbo);
}

void
CPUProcessor::run()
{
  // Draws space-screen quad
  glUseProgram(_shader);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, _screen_tex);
  glBindVertexArray(_quadVao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
