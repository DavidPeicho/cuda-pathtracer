#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/geometric.hpp>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <string>

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

    void upload_camera(struct SceneData &out_scene)
    {
      // DEBUG
      Camera cam;
      cam.position = glm::vec3(20.f,10.f, 20.0f) - glm::vec3(193.f, 86.f, 215.f) / 10.f;
      cam.fov_x = (100.0 * M_PI) / 180.0;
      cam.u[0] = 1.0;
      cam.u[1] = 0.0;
      cam.u[2] = 0.0;
      cam.v[0] = 0.0;
      cam.v[1] = 1.0;
      cam.v[2] = 0.0;
      cam.u = glm::normalize(cam.u);
      cam.v = glm::normalize(cam.v);
      cam.dir = glm::normalize(glm::vec3(0.f, -0.1f, .1f));
      // END DEBUG

      cudaMalloc(&out_scene.cam, sizeof (struct Camera));
      cudaMemcpy(out_scene.cam, &cam, sizeof (struct Camera),
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
                          Buffer<Material> &out_materials)
    {
      size_t nb_mat = materials.size();
      size_t nb_bytes = nb_mat * sizeof(Material);
      Material *tmp_materials = new Material[nb_mat];

      // This is gross, but I did not find a way to copy the
      // data inside the struct directly on the GPU memory.
      for (size_t i = 0; i < nb_mat; ++i)
      {
        Material &mat = tmp_materials[i];
        memcpy(&mat.ambient, &materials[i].ambient, 3 * sizeof (Real));
        memcpy(&mat.diffuse, &materials[i].diffuse, 3 * sizeof(Real));
        memcpy(&mat.specular, &materials[i].specular, 3 * sizeof(Real));
        memcpy(&mat.transmittance, &materials[i].transmittance, 3 * sizeof(Real));
        memcpy(&mat.emission, &materials[i].emission, 3 * sizeof(Real));
        memcpy(&mat.shininess, &materials[i].shininess, sizeof(Real));
      }

      out_materials.size = nb_mat;
      cudaMalloc(&out_materials.data, nb_bytes);
      cudaThrowError();
      cudaMemcpy(out_materials.data, tmp_materials, nb_bytes,
        cudaMemcpyHostToDevice);
      cudaThrowError();

      delete tmp_materials;
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
  { }

  Scene::Scene(const std::string&& filepath)
        : _filepath(filepath)
        , _uploaded(false)
        , _ready{ false }
  { }

  void
  Scene::upload(bool is_cpu)
  {
    if (_uploaded)
      return;

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    tinyobj::attrib_t attrib;

    _ready = tinyobj::LoadObj(&attrib, &shapes,
      &materials, &_load_error, _filepath.c_str());

    if (!_ready)
      return;

    if (is_cpu)
      upload_cpu(shapes, materials, attrib);
    else
      upload_gpu(shapes, materials, attrib);

    _uploaded = true;
  }

  void
  Scene::release(bool is_cpu)
  {
    if (!_uploaded || !_ready)
      return;

    if (is_cpu)
      release_cpu();
    else
      release_gpu();

    _uploaded = false;
  }

  void
  Scene::upload_cpu(const std::vector<tinyobj::shape_t> &shapes,
    const std::vector<tinyobj::material_t>& materials,
    const tinyobj::attrib_t attrib)
  {
    _scene_data = new SceneData;
    _scene_data->cam = new Camera;

    auto &cam = *_scene_data->cam;
    // DEBUG
    cam.position[0] = 0.0;
    cam.position[1] = -3.0;
    cam.position[2] = -5.0;

    cam.fov_x = (90.0 * M_PI) / 180.0;
    cam.u[0] = 1.0;
    cam.u[1] = 0.0;
    cam.u[2] = 0.0;
    cam.v[0] = 0.0;
    cam.v[1] = 1.0;
    cam.v[2] = 0.0;
    cam.u = glm::normalize(cam.u);
    cam.v = glm::normalize(cam.v);
    cam.dir = glm::normalize(glm::cross(cam.v, cam.u));
    // END DEBUG

    auto nb_vertices = attrib.vertices.size();
    auto nb_vertices_bytes = nb_vertices * sizeof(tinyobj::real_t);
    auto nb_normals = attrib.normals.size();
    auto nb_normals_bytes = nb_normals * sizeof(tinyobj::real_t);

    // Uploads attributes & materials
    _scene_data->vertices.size = nb_vertices;
    _scene_data->vertices.data = new tinyobj::real_t[nb_vertices];
    memcpy(_scene_data->vertices.data, &attrib.vertices[0], nb_vertices_bytes);
    _scene_data->normals.size = nb_normals;
    _scene_data->normals.data = new tinyobj::real_t[nb_normals];
    memcpy(_scene_data->normals.data, &attrib.normals[0], nb_normals_bytes);

    // Uploads materials
    _scene_data->materials.size = materials.size();
    _scene_data->materials.data = new Material[materials.size()];
    for (size_t i = 0; i < materials.size(); ++i)
    {
      Material &mat = _scene_data->materials.data[i];
      memcpy(&mat.ambient, &materials[i].ambient, 3 * sizeof(Real));
      memcpy(&mat.diffuse, &materials[i].diffuse, 3 * sizeof(Real));
      memcpy(&mat.specular, &materials[i].specular, 3 * sizeof(Real));
      memcpy(&mat.transmittance, &materials[i].transmittance, 3 * sizeof(Real));
      memcpy(&mat.emission, &materials[i].emission, 3 * sizeof(Real));
      memcpy(&mat.shininess, &materials[i].shininess, sizeof(Real));
    }

    // Uploads meshes
    _scene_data->meshes.size = shapes.size();
    _scene_data->meshes.data = new Mesh[shapes.size()];
    for (size_t i = 0; i < shapes.size(); ++i)
    {
      Mesh &mesh = _scene_data->meshes.data[i];

      mesh.indices.size = shapes[i].mesh.indices.size();
      mesh.indices.data = new tinyobj::index_t[mesh.indices.size];
      auto idx_bytes = mesh.indices.size * sizeof(tinyobj::index_t);
      std::memcpy(mesh.indices.data, &shapes[i].mesh.indices[0], idx_bytes);

      mesh.material_ids.size = shapes[i].mesh.material_ids.size();
      mesh.material_ids.data = new int[mesh.material_ids.size];
      auto mat_bytes = mesh.material_ids.size * sizeof(int);
      memcpy(mesh.material_ids.data, &shapes[i].mesh.material_ids[0], mat_bytes);
    }
  }

  void
  Scene::upload_gpu(const std::vector<tinyobj::shape_t> &shapes,
    const std::vector<tinyobj::material_t>& materials,
    const tinyobj::attrib_t attrib)
  {
    // _sceneData is allocated on the stack,
    // and allows to handle cudaMalloc & cudaFree

    //
    // Lines below copy adresses given by the GPU the stack-allocated
    // SceneData struct.
    // Takes also care of making cudaMemcpy of the data.
    //
    /*upload_camera(_sceneData);
    upload_attribute(_attrib.vertices, _sceneData.vertices);
    upload_attribute(_attrib.normals, _sceneData.normals);
    upload_materials(_materials, _sceneData.materials);
    upload_meshes(_shapes, _sceneData.meshes);
    // Now the sceneData struct contains pointers to memory adresses
    // mapped by the GPU, we can send the whole struct to the GPU.
    cudaMalloc(&_d_sceneData, sizeof(struct SceneData));
    cudaThrowError();
    cudaMemcpy(_d_sceneData, &_sceneData, sizeof(struct SceneData),
      cudaMemcpyHostToDevice);
    cudaThrowError();*/

  }

  void
  Scene::release_cpu()
  {
    delete _scene_data->cam;
    delete _scene_data->meshes.data;
    delete _scene_data->vertices.data;
    delete _scene_data->normals.data;
    delete _scene_data->materials.data;

    delete _scene_data;
  }

  void
  Scene::release_gpu()
  {
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