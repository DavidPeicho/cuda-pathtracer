#pragma once

#include <algorithm>
#include <cstring>

#include <iostream>

#include <glm/common.hpp>

#include <stb/stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include "material_loader.h"

namespace scene
{
  namespace
  {
    Texture
    pack(const Texture& rgb, const Texture& a)
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

    Texture
    pack(const Texture &tex, glm::vec3 default_v)
    {
      Texture result = { tex.w, tex.h, 4, new float[tex.w * tex.h * 4] };

      size_t size = tex.h * tex.w * 4;
      size_t j = 0;
      for (size_t i = 0; i < size; i += 4)
      {
        result.data[i] = default_v[0];
        result.data[i + 1] = default_v[1];
        result.data[i + 2] = default_v[2];
        result.data[i + 3] = tex.data[j++];
      }
      return result;
    }

    /// <summary>
    /// Adds one additional value to a non-null Texture.
    /// </summary>
    /// <param name="tex">Non-null texture containing RGB data.</param>
    /// <param name="default">Default value of the last channel (A channel).</param>
    /// <returns></returns>
    Texture
    pack(const Texture &tex, float default_v)
    {
      Texture result = { tex.w, tex.h, 4, new float[tex.w * tex.h * 4] };

      size_t size = tex.h * tex.w * 4;
      size_t j = 0;
      for (size_t i = 0; i < size; i += 4)
      {
          result.data[i] = tex.data[j];
          result.data[i + 1] = tex.data[j + 1];
          result.data[i + 2] = tex.data[j + 2];
          result.data[i + 3] = default_v;
          j += 3;
      }
      return result;
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
      float *tmp = stbi_loadf(full_path.c_str(), &w, &h, &nb_chan, STBI_default);
      if (tmp == nullptr)
      {
        std::cerr << "arttracer: MaterialLoader: failed to open "
                  << full_path << std::endl;

        loaded_tex[tex] = Texture{ 0, 0, 0, nullptr };
        return;
      }

      // This is actually gross, we duplicate the data in order
      // to free it easily later. If we do not use this, we will
      // be stuck in the dtor to free the textures.
      float *data = new float[w * h * nb_chan];
      std::memcpy(data, tmp, w * h * nb_chan * sizeof(float));

      stbi_image_free(tmp);

      loaded_tex[tex] = Texture{
        w, h, nb_chan, data
      };
    }

    template <typename T>
    Texture
    createUnitTex(T default_v, int nb_chan)
    {
      float *data = new float[nb_chan];
      for (unsigned int i = 0; i < nb_chan; ++i) data[i] = default_v[i];

      return { 1, 1, nb_chan, data };
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

    // Some textures were directly added into the '_textures' vector.
    // These textures were not packed so no intermediary allocation was made.
    // We have to take this into account while freeing the memory.

    for (auto& pair : _loaded_tex)
    {
      // Texture has been loaded but not used
      // directly by itself, it should be freed.
      if (!_packed_tex.count(pair.first)) delete pair.second.data;
    }
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

      glm::vec3 default_diff_rgb(
        tiny_mat.diffuse[0], tiny_mat.diffuse[1], tiny_mat.diffuse[2]
      );
      float default_spec = (
        tiny_mat.specular[0] + tiny_mat.specular[1] + tiny_mat.specular[2]
      ) / 3.0;

      // Loads diffuse and specular map.
      mat.diffuse_spec_map = getTextureId(tiny_mat.diffuse_texname,
        tiny_mat.specular_texname, default_diff_rgb, default_spec);

      // Loads normal map, identified either by `norm' or
      // by `bump'.
      std::string normal_map_path = !tiny_mat.bump_texname.empty() ?
        tiny_mat.bump_texname : tiny_mat.normal_texname;
      mat.normal_map = getTextureId(normal_map_path);

      _materials_gpu.push_back(mat);
    }

    out_tex = _textures;
    out_mat = _materials_gpu;
  }

  int
  MaterialLoader::getTextureId(std::string tex_rgb)
  {
    if (tex_rgb.empty()) return -1;

    return registerOrGet(tex_rgb);
  }

  int
  MaterialLoader::getTextureId(std::string tex_rgb, glm::vec3 default_rgb)
  {
    if (tex_rgb.empty())
    {
      _textures.push_back(createUnitTex<glm::vec3>(default_rgb, 3));
      unsigned int curr_id = ID++;
      return curr_id;
    }
    return registerOrGet(tex_rgb);
  }

  int
  MaterialLoader::getTextureId(std::string tex_rgb, std::string tex_a,
      glm::vec3 default_rgb, float default_a)
  {
    // CASE 1: There is no texture provided.
    // We will build a custom texture if the material
    // is not provided any. We will create a 1x1 pixel-wide texture.
    if (tex_rgb.empty() && tex_a.empty())
    {
      _textures.push_back(createUnitTex<glm::vec4>(
        glm::vec4(default_rgb.r, default_rgb.g, default_rgb.b, default_a), 4
      ));

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
      unsigned curr_id = ID++;
      const Texture &tex = _loaded_tex[tex_a];
      if (tex.nb_chan == 1)
      {
        _textures.push_back(pack(tex, default_rgb));
        return curr_id;
      }

      // The texture loading failed, we will send a unit texture.
      std::cerr << "arttracer: MaterialLoader: \n"
                << "- '" << tex_a << "': invalid nb of channels."
                << std::endl;

      _textures.push_back(createUnitTex<glm::vec4>(
        glm::vec4(default_rgb.r, default_rgb.g, default_rgb.b, default_a), 4
      ));

      return curr_id;
    }

    if (!tex_rgb.empty() && tex_a.empty())
    {
      unsigned curr_id = ID++;
      const Texture &tex = _loaded_tex[tex_rgb];
      if (tex.nb_chan == 3)
      {
        _textures.push_back(pack(tex, default_a));
        return curr_id;
      }

      // The texture loading failed, we will send a unit texture.
      std::cerr << "arttracer: MaterialLoader: \n"
                << "- '" << tex_rgb << "': invalid nb of channels."
                << std::endl;

      _textures.push_back(createUnitTex(
        glm::vec4(default_rgb.r, default_rgb.g, default_rgb.b, default_a), 4
      ));

      return curr_id;
    }

    // CASE 3: Both textures are provided.
    // We can pack them together and cache the result.
    std::string token = tex_rgb + tex_a;
    if (_packed_tex.count(token)) return _packed_tex[token];

    Texture& rgb_tex = _loaded_tex[tex_rgb];
    Texture& a_tex = _loaded_tex[tex_a];

    // One of the textures, or both could not be loaded.
    // We send a packed texture according to which one is not loaded.
    if (rgb_tex.nb_chan != 3 || a_tex.nb_chan != 1)
    {
      unsigned curr_id = ID++;
      // Both failed, we send a unit texture
      if (rgb_tex.nb_chan != 3 && a_tex.nb_chan != 1)
      {
        std::cerr << "arttracer: MaterialLoader: \n"
                  << "- '" << tex_rgb << "' invalid nb of channels.\n"
                  << "- '" << tex_a << "' invalid nb of channels." << std::endl;

        _textures.push_back(createUnitTex(
          glm::vec4(default_rgb.r, default_rgb.g, default_rgb.b, default_a), 4
        ));
        return curr_id;
      }
      // Only one of the two texture fail why loading,
      // we will pack the one that has successfully loaded
      // with the default value of the other.

      // RGB texture failed
      if (rgb_tex.nb_chan != 3)
      {
        std::cerr << "arttracer: MaterialLoader: \n"
          << "- '" << tex_rgb << "': invalid nb of channels."
          << std::endl;

        _textures.push_back(pack(a_tex, default_rgb));
        _packed_tex[token] = curr_id;
        return _packed_tex[token];
      }


      // A texture failed
      std::cerr << "arttracer: MaterialLoader: \n"
                << "- '" << tex_a << "': invalid nb of channels."
                << std::endl;

      _textures.push_back(pack(rgb_tex, default_a));
      _packed_tex[token] = curr_id;
      return _packed_tex[token];
    }

    // Both textures do not have the same size,
    // we will need to scale them accordingly.
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
        delete rgb_tex.data;

        rgb_tex.w = a_tex.w;
        rgb_tex.h = a_tex.h;
        rgb_tex.data = new_data;
      }
      else
      {
        float *new_data = new float[rgb_tex.w * rgb_tex.h * 1];
        stbir_resize_float(a_tex.data, a_tex.w, a_tex.h, 0, new_data, rgb_tex.w, rgb_tex.h, 0, 1);
        delete a_tex.data;

        a_tex.w = a_tex.w;
        a_tex.h = a_tex.h;
        a_tex.data = new_data;
      }
    }

    Texture packed = pack(rgb_tex, a_tex);
    _textures.push_back(packed);
    _packed_tex[token] = ID++;

    return _packed_tex[token];
  }

  int
  MaterialLoader::registerOrGet(std::string tex_rgb)
  {
    if (_packed_tex.count(tex_rgb)) return _packed_tex[tex_rgb];

    checkAndupload(tex_rgb, _mtl_folder, _loaded_tex);

    const Texture& tex = _loaded_tex[tex_rgb];
    if (tex.nb_chan == 0)
      return -1;

    _packed_tex[tex_rgb] = ID;
    _textures.push_back(tex);

    unsigned int curr_id = ID++;
    return curr_id;
  }
}
