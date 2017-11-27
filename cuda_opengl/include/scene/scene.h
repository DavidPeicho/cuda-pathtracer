#pragma once

#include <string>

#include <tiny_obj_loader.h>

#include "scene_data.h"

namespace scene {
/// <summary>
/// Loads primitives of a given scene using the
/// TinyObjLoader library.
///
/// When the scene is uploaded to the GPU, it is
/// released from the CPU to reduce memory overhead.
/// </summary>
class Scene
{
public:
  Scene(const std::string& filepath);
  Scene(const std::string&& filepath);

public:
  /// <summary>
  /// Uploads the CPU scene to the GPU.
  /// </summary>
  /// <param name="camera">Not used anymore.</param>
  void upload(scene::Camera* camera);

  /// <summary>
  /// Releases the scene from the GPU.
  /// </summary>
  void release();

  const inline std::string& getSceneName() { return _filepath; }

  const inline std::string& getCubemapPath() const { return _cubemap_path; }

  const inline scene::Camera& getInitCamera() const { return _init_camera; }

  const inline scene::SceneData* getUploadedScenePointer() const
  {
    return _d_scene_data;
  }

  inline bool ready() { return _ready; }

  inline std::string& error() { return _load_error; }

private:
  void upload_gpu(const std::vector<tinyobj::shape_t>& shapes,
                  const std::vector<tinyobj::material_t>& materials,
                  const tinyobj::attrib_t attrib,
                  const std::string& base_folder);

  void release_gpu();

private:
  std::string _filepath;
  std::string _cubemap_path;

  bool _uploaded;
  bool _ready;
  std::string _load_error;

  /// <summary>
  /// Initial data of the camera.
  /// </summary>
  scene::Camera _init_camera;

  scene::SceneData* _scene_data;
  scene::SceneData* _d_scene_data;
};

} // namespace scene
