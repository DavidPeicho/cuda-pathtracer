#include <cuda.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "driver/cuda_helper.h"

#include "gpu_processor.h"

#include "texture_utils.h"
#include "utils.h"
#include "shaders/raytrace.h"

namespace processor
{
  namespace
  {
    static const float3 WORLD_DOWN_VEC = make_float3(0.0f, -1.0f, 0.0f);

    /// <summary>
    /// Creates a texture of size 1x1, useful when no cubemap is speicified.
    /// THis allows us to use the same code path and without additional performance
    /// downgrade.
    /// </summary>
    /// <param name="nb_chan">The number of channels of the created texture.</param>
    /// <param name="color">The color of the only pixel in the texture, in hexadecimal.</param>
    /// <param name="nb_faces">The number of faces (6 for a cubemap).</param>
    /// <returns>
    ///   A point to the texture if the allocation succeeded, <c>nullptr</c> otherwise.
    /// </returns>
    float *createUnitTexture(unsigned int nb_chan, unsigned int color,
      unsigned int nb_faces)
    {
      static const unsigned int DEFAULT_SIZE = 1;

      float r = ((color >> 16) & 0xFF) / 255.0f;
      float g = ((color >> 8) & 0xFF) / 255.0f;
      float b = (color & 0xFF) / 255.0f;

      unsigned int nb_elt = DEFAULT_SIZE * DEFAULT_SIZE * nb_chan;
      float *img = new float[nb_elt * nb_faces];
      for (unsigned int n = 0; n < nb_faces; ++n)
      {
        for (unsigned int i = 0; i < nb_elt; i += nb_chan)
        {
          img[n * nb_elt + i] = r;
          img[n * nb_elt + i + 1] = g;
          img[n * nb_elt + i + 2] = b;
          img[n * nb_elt + i + 3] = 0.0;
        }
      }
      return img;
    }

    /// <summary>
    /// Uploads a cubemap to the GPU. Only one cubemap can be used
    ///
    /// </summary>
    /// <param name="path">The path.</param>
    /// <param name="out_scene">The out scene.</param>
    void upload_cubemap(const std::string &path, scene::Cubemap& cubemap)
    {
      static const unsigned int NB_COMP = 4;
      static const unsigned int NB_FACES = 6;
      static const unsigned int DEFAULT_COLOR = 0x05070A;

      // This pointer will contain the image data, either loaded
      // on the disk, or created by using a constant color.
      float *img = nullptr;
      unsigned int size = 1;

      // This is used to free the pointer correctly,
      // using either the delete operator, or the
      // stbi_image_free call.
      float *loaded = nullptr;

      // No cubemap was provided with the scene, we will
      // either use a given hexadecimal color, or use the default color.
      if (path.empty() || utils::isHexa(path))
      {
        img = createUnitTexture(
          NB_COMP,
          path.empty() ? DEFAULT_COLOR : strtol(path.c_str(), NULL, 16),
          NB_FACES
        );
      }
      // A path to a cubemap has been provided, we load it and extract
      // each face from the cubecross.
      else
      {
        std::string error;
        int w, h, nb_chan;
        loaded = stbi_loadf(path.c_str(), &w, &h, &nb_chan, STBI_default);

        size = w / 4;

        // An internal error occured in stbi_loadf.
        if (!loaded)
          error = "unknown error " + path;
        // Width and height are not the same, the cubecross can not be valid.
        else if (size != h / 3)
          error = "width and height are not the same";
        // NPOT.
        else if (size & (size - 1))
          error = "size should be a power of 2";

        // No error has occured when loading the cubemap, we
        // can upload it safely.
        if (error.empty())
        {
          // CUDA cubemap textures expect the data to lay out as follows:
          // +x / -x / +y / -y / +z / -z
          img = new float[size * size * NB_COMP * NB_FACES];
          // Sends +/- X faces
          float *tmp = texture::append_cube_faces(
            img, loaded, w, 0, nb_chan, NB_COMP, true, true
          );
          // Sends +/- Y faces
          tmp = texture::append_cube_faces(
            tmp, loaded, w, 1, nb_chan, NB_COMP, false, false
          );
          // Sends +/- Z faces
          texture::append_cube_faces(
            tmp, loaded, w, 1, nb_chan, NB_COMP, true, false
          );
        }
        else
        {
          std::cerr << "arttracer: cubemap loading fail: " << error << std::endl;
          img = createUnitTexture(NB_COMP, DEFAULT_COLOR, NB_FACES);
        }
      }

      cubemap.cubemap_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
      cudaThrowError();

      cudaMalloc3DArray(&cubemap.cubemap, &cubemap.cubemap_desc,
        make_cudaExtent(size, size, NB_FACES), cudaArrayCubemap);
      cudaThrowError();

      cudaMemcpy3DParms myparms = { 0 };
      myparms.srcPos = make_cudaPos(0, 0, 0);
      myparms.dstPos = make_cudaPos(0, 0, 0);
      myparms.srcPtr = make_cudaPitchedPtr(img, size * sizeof(float) * NB_COMP, size, size);
      myparms.dstArray = cubemap.cubemap;
      myparms.extent = make_cudaExtent(size, size, NB_FACES);
      myparms.kind = cudaMemcpyHostToDevice;
      cudaMemcpy3D(&myparms);
      cudaThrowError();

      delete img;
      if (loaded) stbi_image_free(loaded);
    }
  }

  GPUProcessor::GPUProcessor(std::vector<std::string> scene_names,
    std::string cubemap, int width, int height)
               : _scene_names(scene_names)
               , _cubemap_path(cubemap)
               , _scene_id(0)
               , _prev_scene_id(0)
               , _interop(width, height)
               , _d_temporal_framebuffer(nullptr)
               , _moved(false)
  {
    cudaStreamCreateWithFlags(&_stream, cudaStreamDefault);
    cudaMalloc(&_d_temporal_framebuffer, width * height * sizeof(float3));

    // Initializes all keys to released
    for (size_t i = 0; i < 65536; ++i) _keys[i] = false;

    // Creates associated files scenes
    for (const auto &file : scene_names) _scenes.emplace_back(file);
  }

  void
  GPUProcessor::init()
  {
    std::cout << "Uploading scenes to VRAM"
              << ", this operation may take a few seconds."
              << std::endl;

    std::cout << _gpu_info.getProfile() << "\n" << std::endl;

    size_t free_space = 0;
    size_t consumed = 0;
    size_t total_consumed = 0;

    // Uploads scene for GPU usage
    for (auto& scene : _scenes)
    {
      std::cout << "Uploading scene `" << scene.getSceneName()
        << "'..." << std::endl;

      free_space = _gpu_info.getFreeMo();

      scene.upload(&_camera);
      consumed = free_space - _gpu_info.getFreeMo();
      total_consumed += consumed;

      std::cout << consumed << " (MB) uploaded!\n" << std::endl;
    }

    std::cout << "Uploading Cubemap `" << _cubemap_path << "'..." << std::endl;

    upload_cubemap(_cubemap_path, _cubemap);
    consumed = free_space - _gpu_info.getFreeMo();
    total_consumed += consumed;

    std::cout << free_space - _gpu_info.getFreeMo() << " (MB) uploaded!\n" << std::endl;

    // Recaps the overall VRAM consummed. This gives a good idea
    // how streaming is important, especially with textures!
    std::cout << "Uploading completed!\n"
      << "* " << _scenes.size() << " scenes uploaded.\n"
      << "* " << total_consumed << " (MB) VRAM consumed." << std::endl;
  }

  void
  GPUProcessor::update(float delta)
  {
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
    if (_scenes.size() == 0) return;

    if (_prev_scene_id != _scene_id)
    {
      cudaDeviceSynchronize();
      _prev_scene_id = _scene_id;
      std::cout << "toto" << std::endl;
    }

    const auto *gpu_scene = _scenes[_scene_id].getUploadedScenePointer();

    if (gpu_scene == nullptr) return;

    cudaError_t cuda_err = _interop.map(_stream);
    raytrace(
      _interop.getArray(), gpu_scene, _cubemap, &_camera,
      _interop.width(), _interop.height(), _stream,
      _d_temporal_framebuffer, _moved
    );
    cuda_err = _interop.unmap(_stream);

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
    if (k) this->setMoved(true);
    return k;
  }

} // namespace processor
