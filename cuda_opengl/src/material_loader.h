#pragma once

#include <tiny_obj_loader.h>
#include <unordered_map>
#include <vector>

#include "scene/scene_data.h"
#include "shaders/cutils_math.h"

namespace scene
{
  class MaterialLoader
  {
    public:
      MaterialLoader(const std::vector<tinyobj::material_t>& tiny_materials, const std::string mtl_folder);
      ~MaterialLoader();

    public:
      void
      load(std::vector<Texture>&, std::vector<Material>&);

    private:
      int
      getTextureId(std::string tex_rgb);

      int
      getTextureId(std::string tex_rgb, float3 default_rgb);

      int
      getTextureId(std::string tex_rgb, std::string tex_a,
        float3 default_rgb, float default_a);

      int
      registerOrGet(std::string tex_rgb);

    private:
      const std::string _mtl_folder;

      const std::vector<tinyobj::material_t>& _tiny_materials;

      /// <summary>
      /// Contains loaded from disk texture. These textures are not ready yet
      /// to be sent to the GPU. We still have to pack them tight.
      /// </summary>s
      std::unordered_map<std::string, Texture> _loaded_tex;

      /// <summary>
      /// Contains all the textures that have been packed together
      /// </summary>
      std::unordered_map<std::string, int> _packed_tex;

      /// <summary>
      /// Contains the materials to send to the GPU.
      /// </summary>
      std::vector<Material> _materials_gpu;

      /// <summary>
      /// Contains the texture to send to the GPU.
      /// Textures inside this vector are packed together for efficiency.
      /// </summary>
      std::vector<Texture> _textures;
  };
}
