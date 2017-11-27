#include <cuda.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstring>
#include <iostream>
#include <unordered_set>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <driver/cuda_helper.h>

#include <gpu_processor.h>
#include <scene/material_loader.h>
#include <shaders/raytrace.h>
#include <utils/texture_utils.h>
#include <utils/utils.h>

namespace processor {
namespace {
static const float3 WORLD_DOWN_VEC = make_float3(0.0f, -1.0f, 0.0f);

/// <summary>
/// Creates a texture of size 1x1, useful when no cubemap is speicified.
/// THis allows us to use the same code path and without additional performance
/// downgrade.
/// </summary>
/// <param name="nb_chan">The number of channels of the created texture.</param>
/// <param name="color">The color of the only pixel in the texture, in
/// hexadecimal.</param>
/// <param name="nb_faces">The number of faces (6 for a cubemap).</param>
/// <returns>
///   A point to the texture if the allocation succeeded, <c>nullptr</c>
///   otherwise.
/// </returns>
float*
createUnitTexture(unsigned int nb_chan, unsigned int color,
                  unsigned int nb_faces)
{
  static const unsigned int DEFAULT_SIZE = 1;

  float r = ((color >> 16) & 0xFF) / 255.0f;
  float g = ((color >> 8) & 0xFF) / 255.0f;
  float b = (color & 0xFF) / 255.0f;

  unsigned int nb_elt = DEFAULT_SIZE * DEFAULT_SIZE * nb_chan;
  float* img = new float[nb_elt * nb_faces];
  for (unsigned int n = 0; n < nb_faces; ++n) {
    for (unsigned int i = 0; i < nb_elt; i += nb_chan) {
      img[n * nb_elt + i] = r;
      img[n * nb_elt + i + 1] = g;
      img[n * nb_elt + i + 2] = b;
      img[n * nb_elt + i + 3] = 0.0;
    }
  }
  return img;
}

/// <summary>
/// Creates a cubemap and allocates it on the GPU.
/// </summary>
/// <param name="path">The path of the cubemap to load.</param>
/// <returns>
///   Cubemap structure containing the cuda array as well as,
///   the cubemap params.
/// </returns>
scene::Cubemap
uploadCubemap(const std::string& path)
{
  scene::Cubemap cubemap;

  static const unsigned int NB_COMP = 4;
  static const unsigned int NB_FACES = 6;
  static const unsigned int DEFAULT_COLOR = 0x131b23;

  // This pointer will contain the image data, either loaded
  // on the disk, or created by using a constant color.
  float* img = nullptr;
  unsigned int size = 1;

  // This is used to free the pointer correctly,
  // using either the delete operator, or the
  // stbi_image_free call.
  float* loaded = nullptr;

  // No cubemap was provided with the scene, we will
  // either use a given hexadecimal color, or use the default color.
  if (path.empty() || utils::isHexa(path)) {
    img = createUnitTexture(
      NB_COMP, path.empty() ? DEFAULT_COLOR : strtol(path.c_str(), NULL, 16),
      NB_FACES);
  }
  // A path to a cubemap has been provided, we load it and extract
  // each face from the cubecross.
  else {
    std::string error;
    int w, h, nb_chan;
    loaded = stbi_loadf(path.c_str(), &w, &h, &nb_chan, STBI_default);

    size = w / 4;

    // An internal error occured in stbi_loadf.
    if (!loaded)
      error = "unknown error " + path;
    // Width and height are not the same, the cubecross can not be valid.
    else if (size != (unsigned)(h) / 3)
      error = "width and height are not the same";
    // NPOT.
    else if (size & (size - 1))
      error = "size should be a power of 2";

    // No error has occured when loading the cubemap, we
    // can upload it safely.
    if (error.empty()) {
      // CUDA cubemap textures expect the data to lay out as follows:
      // +x / -x / +y / -y / +z / -z
      img = new float[size * size * NB_COMP * NB_FACES];
      // Sends +/- X faces
      float* tmp = texture::append_cube_faces(img, loaded, w, 0, nb_chan,
                                              NB_COMP, true, true);
      // Sends +/- Y faces
      tmp = texture::append_cube_faces(tmp, loaded, w, 1, nb_chan, NB_COMP,
                                       false, false);
      // Sends +/- Z faces
      texture::append_cube_faces(tmp, loaded, w, 1, nb_chan, NB_COMP, true,
                                 false);
    } else {
      std::cerr << "arttracer: cubemap loading fail: " << error << std::endl;
      img = createUnitTexture(NB_COMP, DEFAULT_COLOR, NB_FACES);
      size = 1;
    }
  }

  cubemap.cubemap_desc =
    cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaThrowError();

  cudaMalloc3DArray(&cubemap.cubemap, &cubemap.cubemap_desc,
                    make_cudaExtent(size, size, NB_FACES), cudaArrayCubemap);
  cudaThrowError();

  cudaMemcpy3DParms myparms;
  std::memset(&myparms, 0, sizeof(cudaMemcpy3DParms));
  myparms.srcPos = make_cudaPos(0, 0, 0);
  myparms.dstPos = make_cudaPos(0, 0, 0);
  myparms.srcPtr =
    make_cudaPitchedPtr(img, size * sizeof(float) * NB_COMP, size, size);
  myparms.dstArray = cubemap.cubemap;
  myparms.extent = make_cudaExtent(size, size, NB_FACES);
  myparms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&myparms);
  cudaThrowError();

  delete[] img;

  if (loaded)
    stbi_image_free(loaded);

  return cubemap;
}

void
uploadCubemaps(const std::string& folder,
               const std::vector<scene::Scene>& raw_scenes,
               std::vector<std::string>& out_names,
               std::vector<scene::Cubemap>& out)
{
  std::unordered_set<std::string> set;
  for (const auto& s : raw_scenes) {
    const auto& path = s.getCubemapPath();
    if (path.empty())
      set.insert("empty");
    else
      set.insert(path);
  }

  for (const auto& map : set) {
    out_names.push_back(map);
    out.push_back(uploadCubemap(folder + "/" + map));
  }
}

void
uploadScenes(const std::vector<scene::Scene>& raw_scenes, scene::Scenes& out)
{
  if (raw_scenes.size() == 0)
    return;

  std::vector<const scene::SceneData*> raw_scenes_ptr(raw_scenes.size());
  for (unsigned int i = 0; i < raw_scenes.size(); ++i) {
    const scene::Scene& s = raw_scenes[i];
    raw_scenes_ptr[i] = s.getUploadedScenePointer();
  }

  // Uploads scenes
  size_t bytes_scenes = raw_scenes_ptr.size() * sizeof(scene::SceneData*);
  cudaMalloc(&out.scenes, bytes_scenes);
  cudaThrowError();
  cudaMemcpy(out.scenes, &raw_scenes_ptr[0], bytes_scenes,
             cudaMemcpyHostToDevice);
  cudaThrowError();
}

void
uploadTextures(scene::Scenes& out)
{
  auto* mat_loader = scene::MaterialLoader::instance();
  const auto& textures = mat_loader->getTextures();

  if (textures.size() == 0)
    return;

  std::vector<scene::Texture> gpu_textures(textures.size());

  out.textures.size = textures.size();
  for (size_t i = 0; i < out.textures.size; ++i) {
    const scene::Texture& cpu_tex = textures[i];
    scene::Texture& gpu_tex = gpu_textures[i];

    gpu_tex.w = cpu_tex.w;
    gpu_tex.h = cpu_tex.h;
    gpu_tex.nb_chan = cpu_tex.nb_chan;

    size_t nb_bytes = cpu_tex.w * cpu_tex.h * cpu_tex.nb_chan * sizeof(float);
    cudaMalloc(&gpu_tex.data, nb_bytes);
    cudaThrowError();
    cudaMemcpy(gpu_tex.data, cpu_tex.data, nb_bytes, cudaMemcpyHostToDevice);
    cudaThrowError();
  }

  size_t nb_bytes = out.textures.size * sizeof(scene::Texture);
  cudaMalloc(&out.textures.data, nb_bytes);
  cudaThrowError();
  cudaMemcpy(out.textures.data, &gpu_textures[0], nb_bytes,
             cudaMemcpyHostToDevice);
  cudaThrowError();
}
}

GPUProcessor::GPUProcessor(const std::string& asset,
                           std::vector<std::string> scene_names, int width,
                           int height)
  : _asset_folder(asset)
  , _scene_names(scene_names)
  , _scene_id(0)
  , _prev_scene_id(0)
  , _cubemap_id(0)
  , _post_id(0)
  , _interop(width, height)
  , _d_temporal_framebuffer(nullptr)
  , _moved(false)
{
  cudaStreamCreateWithFlags(&_stream, cudaStreamDefault);
  cudaMalloc(&_d_temporal_framebuffer, width * height * sizeof(float3));

  // Initializes all keys to released
  for (size_t i = 0; i < 65536; ++i)
    _keys[i] = false;

  // Creates associated files scenes
  for (const auto& file : scene_names)
    _raw_scenes.emplace_back(asset + "/" + file);
}

GPUProcessor::~GPUProcessor()
{
  this->release();
}

void
GPUProcessor::init()
{
  std::cout << "Uploading scenes to VRAM"
            << ", this operation may take a few seconds." << std::endl;

  std::cout << _gpu_info.getProfile() << "\n" << std::endl;

  size_t free_space = 0;
  size_t consumed = 0;
  size_t total_consumed = 0;

  // Uploads scene for GPU usage
  for (auto& scene : _raw_scenes) {
    std::cout << "Uploading scene `" << scene.getSceneName()
              << "' primitives..." << std::endl;

    free_space = _gpu_info.getFreeMo();
    scene.upload(nullptr);
    consumed = free_space - _gpu_info.getFreeMo();
    total_consumed += consumed;

    std::cout << consumed << " (MB) uploaded!\n" << std::endl;
  }

  uploadScenes(_raw_scenes, _scenes);

  // Textures upload.
  // This allows to avoid duplicating data.
  std::cout << "Uploading Scenes textures..." << std::endl;
  free_space = _gpu_info.getFreeMo();

  uploadTextures(_scenes);

  consumed = free_space - _gpu_info.getFreeMo();
  total_consumed += consumed;

  std::cout << consumed << " (MB) uploaded!\n" << std::endl;

  // Cubemaps upload.
  std::cout << "Uploading Cubemaps ..." << std::endl;
  free_space = _gpu_info.getFreeMo();

  uploadCubemaps(_asset_folder, _raw_scenes, _cubemap_names, _cubemaps);

  consumed = free_space - _gpu_info.getFreeMo();
  total_consumed += consumed;

  std::cout << consumed << " (MB) uploaded!\n" << std::endl;

  // Recaps the overall VRAM consummed. This gives a good idea
  // how streaming is important, especially with textures!
  std::cout << "Uploading completed!\n"
            << "* " << _raw_scenes.size() << " scenes uploaded.\n"
            << "* " << total_consumed << " (MB) VRAM consumed." << std::endl;

  // Sets the camera to the data
  // extracted from the first scene.
  if (this->_raw_scenes.size())
    _camera = _raw_scenes[0].getInitCamera();
}

void
GPUProcessor::update(float delta)
{
  // Changes the scene if an change happened in the UI.
  if (_prev_scene_id != _scene_id) {
    _prev_scene_id = _scene_id;
    // Resets the camera to the scene data
    const auto& scene_cam = this->_raw_scenes[_scene_id].getInitCamera();
    _camera = scene_cam;
    return;
  }

  // Updates cam rotation
  _camera.u = cross(WORLD_DOWN_VEC, _camera.dir);
  _camera.v = cross(_camera.dir, _camera.u);

  // Updates cam position
  if (this->isKeyPressed(GLFW_KEY_W))
    _camera.position += _camera.dir * _actual_speed * delta;
  if (this->isKeyPressed(GLFW_KEY_S))
    _camera.position -= _camera.dir * _actual_speed * delta;
  if (this->isKeyPressed(GLFW_KEY_A))
    _camera.position -= _camera.u * _actual_speed * delta;
  if (this->isKeyPressed(GLFW_KEY_D))
    _camera.position += _camera.u * _actual_speed * delta;

  if (this->isKeyPressed(GLFW_KEY_LEFT_SHIFT))
    _actual_speed += 0.05;
  else
    _actual_speed = _camera.speed;
}

void
GPUProcessor::render()
{
  _interop.clear();
  if (_raw_scenes.size() == 0)
    return;

  _interop.map(_stream);
  cudaCheckError();

  raytrace(_interop.getArray(), _scenes, _scene_id, _cubemaps, _cubemap_id,
           &_camera, _interop.width(), _interop.height(), _stream,
           _d_temporal_framebuffer, _moved, _post_id);

  _interop.unmap(_stream);
  cudaCheckError();

  this->setMoved(false);

  _interop.blit();
  _interop.swap();
}

void
GPUProcessor::resize(unsigned int w, unsigned int h)
{
  _interop.setSize(w, h);

  if (_d_temporal_framebuffer != nullptr)
    cudaFree(_d_temporal_framebuffer);

  cudaMalloc(&_d_temporal_framebuffer, h * w * sizeof(float3));
}

void
GPUProcessor::setKeyState(const unsigned int key, bool state)
{
  _keys[key] = state;
}

bool
GPUProcessor::isKeyPressed(const unsigned int key)
{
  bool k = _keys[key];
  if (k)
    this->setMoved(true);
  return k;
}

void
GPUProcessor::release()
{
  // Releases all the scenes.
  for (auto& scene : _raw_scenes) scene.release();
  cudaFree(_scenes.scenes);

  // Releases the textures
  size_t nb_tex = _scenes.textures.size;
  scene::Texture *textures = new scene::Texture[nb_tex];
  cudaMemcpy(
    textures, _scenes.textures.data,
    nb_tex * sizeof(scene::Texture), cudaMemcpyDeviceToHost
  );
  cudaCheckError();

  for (size_t i = 0; i < nb_tex; ++i) cudaFree(textures[i].data);
  delete[] textures;

  // Releases the cubemaps
  for (const auto& cubemap : _cubemaps) cudaFreeArray(cubemap.cubemap);
  _cubemaps.clear();

  cudaFree(_d_temporal_framebuffer);

  // Releases CPU memory
  scene::MaterialLoader::instance()->release();
}

} // namespace processor
