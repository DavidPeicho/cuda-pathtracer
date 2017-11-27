#include <cstring>

namespace texture {
namespace {
inline float*
copy_face(float* dst, const float* cubecross, unsigned int img_width,
          unsigned int x_offset, unsigned int y_offset, unsigned in_nb_comp,
          unsigned int out_nb_comp)
{
  unsigned int size = img_width / 4;
  for (unsigned int y = 0; y < size; ++y) {
    unsigned int yy = y + y_offset;
    for (unsigned x = 0; x < size; ++x) {
      unsigned xx = x + x_offset;
      const float* src = cubecross + (yy * img_width + xx) * in_nb_comp;
      dst[0] = src[0];
      dst[1] = src[1];
      dst[2] = src[2];
      dst[3] = 0.0;
      dst += out_nb_comp;
    }
  }
  return dst;
}
}

float*
append_cube_faces(float* dst, const float* face, unsigned int width,
                  unsigned int x_start, unsigned int in_nb_comp,
                  unsigned int out_nb_comp, bool horizontal, bool reverse_order)
{
  unsigned int size = width / 4;

  // X fetch for appending
  if (horizontal) {
    unsigned diff = 2 * size;
    unsigned x = x_start * size;
    if (reverse_order) {
      x = (x_start + 2) * size;
      diff = -2 * size;
    }
    dst = copy_face(dst, face, width, x, size, in_nb_comp, out_nb_comp);
    dst = copy_face(dst, face, width, x + diff, size, in_nb_comp, out_nb_comp);
    return dst;
  }

  // Y fetch for appending
  dst = copy_face(dst, face, width, x_start * size, 0, in_nb_comp, out_nb_comp);
  dst = copy_face(dst, face, width, x_start * size, 2 * size, in_nb_comp,
                  out_nb_comp);
  return dst;
}
}
