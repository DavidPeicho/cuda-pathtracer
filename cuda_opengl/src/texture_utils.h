#pragma once

namespace texture
{
  float *
  append_cube_faces(float *out, const float *face, unsigned int width,
    unsigned int x_start, unsigned int in_nb_comp, unsigned int out_nb_comp,
    bool horizontal);
}
