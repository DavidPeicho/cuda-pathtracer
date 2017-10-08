#include <cuda.h>
#include <cuda_runtime.h>
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

    using Material = struct scene::Material;
    using Mesh = struct scene::Mesh;
    
    void upload_camera(struct SceneData &out_scene)
    {
      cudaMalloc(&out_scene.cam, sizeof (struct Camera));
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
                          struct Buffer<Material> &out_materials)
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

      delete tmp_materials;
    }

    void upload_meshes(const ShapeVector& shapes,
      struct Buffer<Mesh> &out_meshes)
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

        cudaMalloc(&tmp_mesh.indices.data, nb_bytes_indices);
        cudaThrowError();
        cudaMalloc(&tmp_mesh.material_ids.data, nb_bytes_materials);
        cudaThrowError();

        // Copies `indices' to the Mesh struct
        cudaMemcpy(tmp_mesh.indices.data, &mesh.indices[0], nb_bytes_indices,
          cudaMemcpyHostToDevice);
        // Copies `material_ids'
        cudaMemcpy(&tmp_mesh.material_ids, &mesh.material_ids[0],
          nb_bytes_materials, cudaMemcpyHostToDevice);
      }
      
      out_meshes.size = nb_meshes;
      cudaMalloc(&out_meshes.data, nb_bytes);
      cudaThrowError();
      cudaMemcpy(out_meshes.data, tmp_meshes, nb_bytes, cudaMemcpyHostToDevice);

      delete tmp_meshes;
    }
    
  }

  Scene::Scene(const std::string& filepath)
        : _ready{ false }
  {
    init(filepath.c_str());
  }

  Scene::Scene(const std::string&& filepath)
        : _ready{ false }
  {
    init(filepath.c_str());
  }

  Scene::~Scene()
  {
    release();
  }

  void 
  Scene::upload()
  {
    // _sceneData is allocated on the stack,
    // and allows to handle cudaMalloc & cudaFree

    // 
    // Lines below copy adresses given by the GPU the stack-allocated
    // SceneData struct.
    // Takes also care of making cudaMemcpy of the data.
    // 
    upload_camera(_sceneData);
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

    _uploaded = true;
  }

  void
  Scene::release()
  {
    if (!_uploaded)
      return;
    
    // Free one-depth pointer, saved on the host stack.
    cudaFree(_sceneData.cam);
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

    _uploaded = false;
  }

  void
  Scene::init(const char *filepath)
  { 
    _ready = tinyobj::LoadObj(&_attrib, &_shapes,
                              &_materials, &_load_error, filepath);
  }

} // namespace scene