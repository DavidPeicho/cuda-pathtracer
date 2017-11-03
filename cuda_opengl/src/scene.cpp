#include <algorithm>

#include <ctype.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/geometric.hpp>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <sstream>
#include <string>

#include <unordered_map>

#include "driver/cuda_helper.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "scene.h"

namespace scene
{
  namespace
  {
    using ShapeVector = std::vector<tinyobj::shape_t>;
    using MaterialVector = std::vector<tinyobj::material_t>;
    using Real = tinyobj::real_t;

    bool parse_double3(glm::vec3 &out, std::stringstream& iss)
    {
      if (iss.peek() == std::char_traits<char>::eof())
        return false;

      std::string token;

      if ((iss >> out.x).peek() == std::char_traits<char>::eof())
        return false;
      if ((iss >> out.y).peek() == std::char_traits<char>::eof())
        return false;

      iss >> out.z;
      return true;
    }

    bool parse_double3(tinyobj::real_t out[3], std::stringstream& iss)
    {
      if (iss.peek() == std::char_traits<char>::eof())
        return false;

      std::string token;

      if ((iss >> out[0]).peek() == std::char_traits<char>::eof())
        return false;
      if ((iss >> out[1]).peek() == std::char_traits<char>::eof())
        return false;

      iss >> out[2];
      return true;
    }

    bool parse_camera(scene::Camera &cam, std::stringstream& iss)
    {
      if (!parse_double3(cam.position, iss)) return false;
      if (!parse_double3(cam.u, iss)) return false;
      if (!parse_double3(cam.v, iss)) return false;

      if (iss.peek() == std::char_traits<char>::eof()) return false;

      cam.u = glm::normalize(cam.u);
      cam.v = glm::normalize(cam.v);

      iss >> cam.fov_x;
      cam.fov_x = (cam.fov_x * M_PI) / 180.0;
      cam.dir = glm::cross(cam.u, cam.v);
      return true;
    }

    void parse_scene(std::string filename, scene::SceneData &out_scene,
      scene::Camera &cam, std::string& objfile)
    {
      std::ifstream file;
      file.open(filename);
      if (!file.is_open())
        throw std::runtime_error("parse_scene(): failed to open '" + filename + "'");;

      // Contains every lights. Because we do not use vector
      // on CUDA, we will need to make a deep copy of it.
      std::vector<LightProp> light_vec;

      std::string line;
      std::string token;
      while (std::getline(file, line))
      {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream iss(line);
        iss >> token;

        if (token == "p_light")
        {
          LightProp plight;
          if (!parse_double3(plight.vec, iss))
            throw std::runtime_error("parse_scene(): error parsing p_light vector.");
          if (!parse_double3(plight.color, iss))
            throw std::runtime_error("parse_scene(): error parsing p_light color.");
          if (iss.peek() == std::char_traits<char>::eof())
            throw std::runtime_error("parse_scene(): error parsing p_light emission.");
          iss >> plight.emission;
          if (iss.peek() == std::char_traits<char>::eof())
            throw std::runtime_error("parse_scene(): error parsing p_light radius.");
          iss >> plight.radius;
          light_vec.push_back(plight);
        }
        else if (token == "scene")
        {
          if (iss.peek() == std::char_traits<char>::eof())
            throw std::runtime_error("parse_scene(): error parsing scene file name.");
          iss >> objfile;
        }
        else if (token == "camera")
        {
          if (!parse_camera(cam, iss))
            throw std::runtime_error("parse_scene(): error parsing the camera.");
        }
      }

      if (light_vec.size() == 0)
        return;

      // Copies lights back to CUDA
      const LightProp *lights = &light_vec[0];
      size_t nb_bytes_lights = light_vec.size() * sizeof(LightProp);
      out_scene.lights.size = light_vec.size();
      cudaMalloc(&out_scene.lights.data, nb_bytes_lights);
      cudaThrowError();
      cudaMemcpy(out_scene.lights.data, lights, nb_bytes_lights,
         cudaMemcpyHostToDevice);
      cudaThrowError();
    }

    void upload_attribute(const std::vector<Real> &attribute,
      struct Buffer<Real>& out_buffer)
    {
      size_t nb_bytes = attribute.size() * sizeof(Real);

      out_buffer.size = attribute.size();

      cudaMalloc(&out_buffer.data, nb_bytes);
      cudaThrowError();
      cudaMemcpy(out_buffer.data, &attribute[0], nb_bytes,
        cudaMemcpyHostToDevice);
    }

    /// <summary>
    /// Uploads every materials.
    /// </summary>
    /// <param name="materials">Materials obtained from TinyObjLoader.</param>
    /// <param name="d_materials">Materials storage for the GPU.</param>
    void upload_materials(const MaterialVector &materials,
                          scene::SceneData *scene,
                          const std::string& base_folder)
    {
      Buffer<scene::Material> &out_materials = scene->materials;
      // The 'tex_id' and 'tex_map' variables allows to associate
      // to each texture a unique ID. If several materials share a texture, they
      // will share the same ID.
      // Texture will be loaded at the end, by looping through the map.
      static int tex_id = 0;
      std::unordered_map<std::string, int> tex_map;

      // The function below handles the texture map. If the texture already
      // exists, it simply use the predefined id. If it does not exist yet,
      // it creates a new id.
      auto registerOrGetTexture = [&tex_map](const std::string& tex_name)
      {
        if (tex_name.empty()) return -1;

        if (!tex_map.count(tex_name))
          tex_map[tex_name] = tex_id++;

        return tex_map[tex_name];
      };

      size_t nb_mat = materials.size();
      size_t nb_bytes = nb_mat * sizeof(Material);
      Material *tmp_materials = new Material[nb_mat];

      // This is gross, but I did not find a way to copy the
      // data inside the struct directly on the GPU memory.
      for (size_t i = 0; i < nb_mat; ++i)
      {
        const auto &tiny_mat = materials[i];
        Material &mat = tmp_materials[i];
        memcpy(&mat.ambient, &tiny_mat.ambient, 3 * sizeof (Real));
        memcpy(&mat.diffuse, &tiny_mat.diffuse, 3 * sizeof(Real));
        memcpy(&mat.specular, &tiny_mat.specular, 3 * sizeof(Real));
        memcpy(&mat.transmittance, &tiny_mat.transmittance, 3 * sizeof(Real));
        memcpy(&mat.emission, &tiny_mat.emission, 3 * sizeof(Real));
        memcpy(&mat.shininess, &tiny_mat.shininess, sizeof(Real));
        mat.diffuse_map = registerOrGetTexture(tiny_mat.diffuse_texname);
        mat.spec_map = registerOrGetTexture(tiny_mat.specular_texname);
      }

      out_materials.size = nb_mat;
      cudaMalloc(&out_materials.data, nb_bytes);
      cudaThrowError();
      cudaMemcpy(out_materials.data, tmp_materials, nb_bytes,
        cudaMemcpyHostToDevice);
      cudaThrowError();

      // Deletes temporary materials after they have been sent
      // to the GPU.
      delete tmp_materials;

      // We know all the textures we have to load, it is time
      // to load them.
      std::vector<scene::Texture> textures(tex_map.size());

      // TODO: Pack some textures together, such as spec map + normal_map, etc...
      std::for_each(std::begin(tex_map), std::end(tex_map),
      [&base_folder, &textures](const std::pair<std::string, int>& pair)
      {
        std::string full_path = base_folder + "/" + pair.first;
        int w = 0;
        int h = 0;
        int nb_chan = 0;
        float *img = stbi_loadf(full_path.c_str(), &w, &h, &nb_chan, STBI_default);

        scene::Texture &texture = textures[pair.second];

        texture.w = w;
        texture.h = h;
        cudaMalloc(&texture.data, nb_chan * w * h * sizeof(float));
        cudaThrowError();
        cudaMemcpy(texture.data, img, nb_chan * w * h * sizeof(float),
          cudaMemcpyHostToDevice);
        // When the texture is upload, it can be freed from
        // the RAM.
        stbi_image_free(img);
      });

      // Sends texture to the GPU.
      if (textures.size())
      {
        scene->textures.size = textures.size();
        size_t nb_tex_bytes = sizeof(scene::Texture) * textures.size();
        cudaMalloc(&scene->textures.data, nb_tex_bytes);
        cudaThrowError();
        cudaMemcpy(scene->textures.data, &textures[0], nb_tex_bytes,
          cudaMemcpyHostToDevice);
      }

      tex_id = 0;
    }

    void upload_meshes(const ShapeVector &shapes, Buffer<Mesh> &out_meshes)
    {
      /// The Mesh structure looks like:
      /// {
      ///    tinyobj::index_t *indices;
      ///    size_t nb_indices;
      ///    int *material_ids;
      /// }
      /// `indices' and `material_idx' should also be allocated.

      size_t nb_meshes = shapes.size();
      size_t nb_bytes = nb_meshes * sizeof (Mesh);
      Mesh *tmp_meshes = new Mesh[nb_meshes];

      for (size_t i = 0; i < nb_meshes; ++i)
      {
        auto& mesh = shapes[i].mesh;

        auto nb_indices = mesh.indices.size();
        auto nb_mat = mesh.material_ids.size();
        auto nb_bytes_indices = nb_indices * sizeof(tinyobj::index_t);
        auto nb_bytes_materials = nb_mat * sizeof(int);

        Mesh &tmp_mesh = tmp_meshes[i];
        tmp_mesh.indices.size = nb_indices;
        tmp_mesh.material_ids.size = nb_mat;

        cudaMalloc(&(tmp_mesh.indices.data), nb_bytes_indices);
        cudaThrowError();
        cudaMalloc(&(tmp_mesh.material_ids.data), nb_bytes_materials);
        cudaThrowError();

        // Copies `indices' to the Mesh struct
        cudaMemcpy(tmp_mesh.indices.data, &mesh.indices[0], nb_bytes_indices,
          cudaMemcpyHostToDevice);
        cudaThrowError();
        // Copies `material_ids'
        cudaMemcpy(tmp_mesh.material_ids.data, &mesh.material_ids[0],
          nb_bytes_materials, cudaMemcpyHostToDevice);
        cudaThrowError();
      }

      out_meshes.size = nb_meshes;
      cudaMalloc(&out_meshes.data, nb_bytes);
      cudaThrowError();
      cudaMemcpy(out_meshes.data, tmp_meshes, nb_bytes,
        cudaMemcpyHostToDevice);
      cudaThrowError();

      delete tmp_meshes;
    }

  }

  Scene::Scene(const std::string& filepath)
        : _filepath(filepath)
        , _uploaded(false)
        , _ready(false)
        , _d_scene_data(nullptr)
  { }

  Scene::Scene(const std::string&& filepath)
        : _filepath(filepath)
        , _uploaded(false)
        , _ready{ false }
        , _d_scene_data(nullptr)
  { }

  void
  Scene::upload(bool is_cpu)
  {
    if (_uploaded)
      return;

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    tinyobj::attrib_t attrib;

    // _sceneData is allocated on the heap,
    // and allows to handle cudaMalloc & cudaFree
    _scene_data = new SceneData;
    _camera = new Camera;

    std::string objfilepath;
    std::string base_dir = "";
    std::string mtl_dir = "";
    std::string full_obj_path = "";

    parse_scene(_filepath, *_scene_data, *_camera, objfilepath);

    std::string::size_type pos = _filepath.find_last_of('/');
    if (pos != std::string::npos)
    {
      base_dir = _filepath.substr(0, pos) + "/";
      mtl_dir = base_dir;
      full_obj_path = base_dir + objfilepath;
    }

    // Extracts basedir to find MTL if any.
    pos = objfilepath.find_last_of('/');
    if (pos != std::string::npos)
    {
      mtl_dir = base_dir + "/" + objfilepath.substr(0, pos) + "/";
    }

    _ready = tinyobj::LoadObj(&attrib, &shapes,
        &materials, &_load_error, full_obj_path.c_str(), mtl_dir.c_str());

    if (!_ready)
    {
      delete _scene_data;
      delete _camera;
      return;
    }

    upload_gpu(shapes, materials, attrib, mtl_dir);
    _uploaded = true;
  }

  void
  Scene::release(bool is_cpu)
  {
    if (!_uploaded || !_ready)
      return;

    release_gpu();
    _uploaded = false;
  }

  void
  Scene::upload_gpu(const std::vector<tinyobj::shape_t> &shapes,
    const std::vector<tinyobj::material_t>& materials,
    const tinyobj::attrib_t attrib,
    const std::string& base_folder)
  {
    //
    // Lines below copy adresses given by the GPU the stack-allocated
    // SceneData struct.
    // Takes also care of making cudaMemcpy of the data.
    //
    upload_attribute(attrib.vertices, _scene_data->vertices);
    upload_attribute(attrib.normals, _scene_data->normals);
    upload_attribute(attrib.texcoords, _scene_data->texcoords);
    upload_materials(materials, _scene_data, base_folder);
    upload_meshes(shapes, _scene_data->meshes);

    // Now the sceneData struct contains pointers to memory adresses
    // mapped by the GPU, we can send the whole struct to the GPU.
    cudaMalloc(&_d_scene_data, sizeof(struct SceneData));
    cudaThrowError();
    cudaMemcpy(_d_scene_data, _scene_data, sizeof(struct SceneData),
      cudaMemcpyHostToDevice);
    cudaThrowError();
  }

  void
  Scene::release_gpu()
  {
    delete _camera;
    delete _scene_data->lights.data;

    delete _scene_data;

    // Free one-depth pointer, saved on the host stack.
    /*cudaFree(_sceneData.cam);
    cudaFree(_sceneData.vertices.data);
    cudaFree(_sceneData.normals.data);
    cudaFree(_sceneData.materials.data);

    size_t nb_meshes = _sceneData.meshes.size;
    Mesh *meshes_to_free = new Mesh[nb_meshes];

    // Copies back from GPU all meshes, which have pointers to
    // allocated memory blocks.
    cudaMemcpy(meshes_to_free, _sceneData.meshes.data, nb_meshes * sizeof(Mesh),
      cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nb_meshes; ++i)
    {
      cudaFree(meshes_to_free->indices.data);
      cudaFree(meshes_to_free->material_ids.data);
    }
    cudaFree(_d_sceneData->meshes.data);

    _uploaded = false;*/
  }

} // namespace scene