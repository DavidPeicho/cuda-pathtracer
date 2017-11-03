#pragma once

#include <algorithm>

#include <glm/common.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include "material_loader.h"

namespace scene
{
  namespace
  {
    Texture
    packTexture(const Texture& rgb, const Texture& a)
    {
      size_t size = rgb.w * rgb.h * 4;

      Texture res;
      res.w = rgb.w;
      res.h = rgb.h;
      res.nb_chan = 4;
      res.data = new float[size];

      size_t j = 0;
      for (size_t i = 0; i < size; i += 4)
      {
        res.data[i] = rgb.data[j];
        res.data[i + 1] = rgb.data[j + 1];
        res.data[i + 2] = rgb.data[j + 2];
        res.data[i + 3] = a.data[i / 4];
        j += 3;
      }
      return res;
    }

    void
    checkAndupload(const std::string& tex, const std::string& base_folder,
        std::unordered_map<std::string, Texture>& loaded_tex)
    {
      if (tex.empty()) return;

      if (loaded_tex.count(tex)) return;

      std::string full_path = base_folder + "/" + tex;
      int w = 0;
      int h = 0;
      int nb_chan = 0;
      float *data = stbi_loadf(full_path.c_str(), &w, &h, &nb_chan, STBI_default);
      loaded_tex[tex] = Texture{
        w, h, nb_chan, data
      };
    }

  }

  static unsigned int ID = 0;

  MaterialLoader::MaterialLoader(const std::vector<tinyobj::material_t>& materials,
    const std::string mtl_folder)
    : _mtl_folder(mtl_folder)
    , _tiny_materials(materials)
  { }

  MaterialLoader::~MaterialLoader()
  {
    size_t s = _textures.size();
    for (size_t i = 0; i < s; ++i) delete _textures[i].data;

    //for (auto& pair : _loaded_tex) stbi_image_free(pair.second.data);
  }

  void
  MaterialLoader::load(std::vector<Texture> &out_tex,
    std::vector<Material>& out_mat)
  {
    size_t nb_mat = _tiny_materials.size();
    for (size_t i = 0; i < nb_mat; ++i)
    {
      const tinyobj::material_t &tiny_mat = _tiny_materials[i];
      Material mat;

      glm::vec3 default_diff_rgb(tiny_mat.diffuse[0], tiny_mat.diffuse[1], tiny_mat.diffuse[2]);
      float default_spec = tiny_mat.shininess;

      mat.diffuse_spec_map = getTextureId(tiny_mat.diffuse_texname,
        tiny_mat.specular_texname, default_diff_rgb, default_spec);

      _materials_gpu.push_back(mat);
    }

    out_tex = _textures;
    out_mat = _materials_gpu;

  }

  unsigned int
  MaterialLoader::getTextureId(std::string tex_rgb, glm::vec3 default_rgb)
  {
    if (tex_rgb.empty())
    {
      float *data = new float[3];
      data[0] = default_rgb[0];
      data[1] = default_rgb[1];
      data[2] = default_rgb[2];

      Texture tex = { 1, 1, 3, data };
      _textures.push_back(tex);

      unsigned int curr_id = ID++;
      return curr_id;
    }

    if (_packed_tex.count(tex_rgb)) return _packed_tex[tex_rgb];

    checkAndupload(tex_rgb, _mtl_folder, _loaded_tex);

    const Texture& tex = _loaded_tex[tex_rgb];
    _packed_tex[tex_rgb] = ID;

    _textures.push_back(tex);

    unsigned int curr_id = ID++;
    return curr_id;
  }

  unsigned int
  MaterialLoader::getTextureId(std::string tex_rgb, std::string tex_a,
      glm::vec3 default_rgb, float default_a)
  {
    // CASE 1: There is no texture provided.
    // We will build a custom texture if the material
    // is not provided any. We will create a 1x1 pixel-wide texture.
    if (tex_rgb.empty() && tex_a.empty())
    {
      float *data = new float[4];
      data[0] = default_rgb[0];
      data[1] = default_rgb[1];
      data[2] = default_rgb[2];
      data[3] = default_a;

      Texture tex = { 1, 1, 4, data };
      _textures.push_back(tex);

      unsigned int curr_id = ID++;
      return curr_id;
    }

    checkAndupload(tex_rgb, _mtl_folder, _loaded_tex);
    checkAndupload(tex_a, _mtl_folder, _loaded_tex);

    // CASE 2: One of the texture is not provided.
    // We will duplicate the data, unfortunately, I did
    // not find a better way to handle this.

    if (tex_rgb.empty() && !tex_a.empty())
    {
      const Texture &tex = _loaded_tex[tex_a];
      Texture result = { tex.w, tex.h, 4, new float[tex.w * tex.h * 4] };

      size_t j = 0;
      size_t size = tex.h * tex.w * 4;
      for (size_t i = 0; i < size; i += 4)
      {
        result.data[i] = default_rgb[0];
        result.data[i + 1] = default_rgb[1];
        result.data[i + 2] = default_rgb[2];
        result.data[i + 3] = tex.data[j++];
      }

      _textures.push_back(result);
      unsigned curr_id = ID++;
      return curr_id;
    }

    if (!tex_rgb.empty() && tex_a.empty())
    {
      const Texture &tex = _loaded_tex[tex_rgb];
      Texture result = { tex.w, tex.h, 4, new float[tex.w * tex.h * 4] };

      size_t size = tex.h * tex.w * 4;
      size_t j = 0;
      for (size_t i = 0; i < size; i += 4)
      {
        result.data[i] = tex.data[j];
        result.data[i + 1] = tex.data[j + 1];
        result.data[i + 2] = tex.data[j + 2];
        result.data[i + 3] = default_a;
        j += 3;
      }

      _textures.push_back(result);
      unsigned curr_id = ID++;
      return curr_id;
    }

    // CASE 3: Both textures are provided.
    // We can pack them together and cache the result.
    std::string token = tex_rgb + tex_a;
    if (_packed_tex.count(token)) return _packed_tex[token];

    Texture& rgb_tex = _loaded_tex[tex_rgb];
    Texture& a_tex = _loaded_tex[tex_a];

    // Both textures have the same size, we can
    // safely create the strided new texture.
    if (rgb_tex.w != a_tex.w || rgb_tex.h != a_tex.h)
    {
      // We will choose to resize the texture to
      // the same size, by upscaling the smallest one to the greatest.
      int rgb_size = rgb_tex.w * rgb_tex.h;
      int a_size = a_tex.w * a_tex.h;

      if (a_size > rgb_size)
      {
        float *new_data = new float[a_tex.w * a_tex.h * 3];
        stbir_resize_float(rgb_tex.data, rgb_tex.w, rgb_tex.h, 0, new_data, a_tex.w, a_tex.h, 0, 3);
        rgb_tex.w = a_tex.w;
        rgb_tex.h = a_tex.h;
        rgb_tex.data = new_data;
      }
      else
      {
        float *new_data = new float[rgb_tex.w * rgb_tex.h * 1];
        stbir_resize_float(a_tex.data, a_tex.w, a_tex.h, 0, new_data, rgb_tex.w, rgb_tex.h, 0, 1);
        a_tex.w = a_tex.w;
        a_tex.h = a_tex.h;
        a_tex.data = new_data;
      }
    }

    Texture packed = packTexture(rgb_tex, a_tex);
    _textures.push_back(packed);
    _packed_tex[token] = ID++;

    return _packed_tex[token];
  }
}
