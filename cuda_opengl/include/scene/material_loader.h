#pragma once

#include <tiny_obj_loader.h>
#include <unordered_map>
#include <vector>

#include "scene/scene_data.h"
#include "shaders/cutils_math.h"

namespace scene
{
  /// <summary>
  /// Singleton loading materials. This is used to keep
  /// track of all the texture loaded, pack them, and
  /// improve memory usage.
  ///
  /// Whenever a texture is already loaded, it will not
  /// be loaded again.
  /// </summary>
  class MaterialLoader
  {
    public:
      static inline MaterialLoader*
      instance()
      {
        if (inst == nullptr)
          inst = new MaterialLoader();

        return inst;
      }

    public:
      /// <summary>
      /// Sets data to use for the next loading.
      /// </summary>
      /// <param name="tiny_materials">TinyObjLoader materials</param>
      /// <param name="mtl_folder">Base folder containing the assets (MTL, etc..)</param>
      /// <returns></returns>
      MaterialLoader*
      set(const std::vector<tinyobj::material_t> *tiny_materials,
          const std::string mtl_folder);

      /// <summary>
      /// Loads the data passed to the `set()' method and
      /// returns the associated material.
      /// </summary>
      /// <param name="">Contains the loaded material, ready to be upload to the GPU.</param>
      void
      load(std::vector<Material>&);

      /// <summary>
      /// Releases CPU texture memory.
      /// </summary>
      void
      release();

      inline const std::vector<scene::Texture>&
      getTextures() const
      {
        return _textures;
      }

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
      static MaterialLoader *inst;

    private:
      const std::vector<tinyobj::material_t> *_tiny_materials;
      std::string _mtl_folder;

      unsigned int _id = 0;

      /// <summary>
      /// Contains loaded from disk texture. These textures are not ready yet
      /// to be sent to the GPU. We still have to pack them tight.
      /// </summary>
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
