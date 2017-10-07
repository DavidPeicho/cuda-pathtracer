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
       
    /// <summary>
    /// Uploads the vertices and normals of the complete scene.
    /// This function uploads a contiguous array of the vertices of all meshes.
    /// </summary>
    /// <param name="attrib">Attribute obtained from TinyObjLoader</param>
    /// <param name="d_vertices">Vertices storage for the GPU allocated memory</param>
    /// <param name="d_normals">Normals storage for the GPU allocated memory</param>
    void upload_vertices_normals(const tinyobj::attrib_t &attrib,
                                 Real **d_vertices, Real **d_normals)
    {
      size_t nb_bytes_vertices = attrib.vertices.size() * sizeof(tinyobj::real_t);
      size_t nb_bytes_normals = attrib.normals.size() * sizeof(tinyobj::real_t);

      cudaMalloc(d_vertices, nb_bytes_vertices);
      cudaThrowError();
      cudaMalloc(d_normals, nb_bytes_normals);
      cudaThrowError();
      cudaMemcpy(d_vertices, &attrib.vertices[0], nb_bytes_vertices,
        cudaMemcpyHostToDevice);
      cudaMemcpy(d_normals, &attrib.normals[0], nb_bytes_normals,
        cudaMemcpyHostToDevice);
    }
    
    /// <summary>
    /// Uploads every materials.
    /// </summary>
    /// <param name="materials">Materials obtained from TinyObjLoader.</param>
    /// <param name="d_materials">Materials storage for the GPU.</param>
    void upload_materials(const MaterialVector& materials,
                          Material **d_materials)
    {
      size_t nb_mat = materials.size();
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

      cudaMalloc(d_materials, nb_mat * sizeof(Material));
      cudaThrowError();
      cudaMemcpy(*d_materials, tmp_materials, nb_mat * sizeof (Material),
        cudaMemcpyHostToDevice);

      delete tmp_materials;
    }

    /// <summary>
    /// Converts meshes of type `ShapeVector' to meshes of type `Mesh'.
    /// </summary>
    /// <param name="shapes">Scene shapes obtained with TinyObjLoader.</param>
    /// <param name="d_meshes">Meshes storage for the GPU allocated memory.</param>
    void upload_meshes(const ShapeVector& shapes, Mesh **d_meshes)
    {
      /// The Mesh structure looks like:
      /// {
      ///    tinyobj::index_t *indices;
      ///    size_t nb_indices;
      ///    int *material_ids;
      /// }
      /// `indices' and `material_idx' should also be allocated.

      size_t nb_meshes = shapes.size();
      Mesh *tmp_meshes = new Mesh[nb_meshes];

      for (size_t i = 0; i < nb_meshes; ++i)
      {
        auto& mesh = shapes[i].mesh;
        auto nb_indices = mesh.indices.size();
        auto nb_bytes_indices = nb_indices * sizeof(tinyobj::index_t);
        auto nb_bytes_materials = mesh.material_ids.size() * sizeof(int);

        Mesh &tmp_mesh = tmp_meshes[i];

        cudaMalloc(&tmp_mesh.indices, nb_bytes_indices);
        cudaThrowError();
        cudaMalloc(&tmp_mesh.material_ids, nb_bytes_materials);
        cudaThrowError();

        // Copies `indices' to the Mesh struct
        cudaMemcpy(tmp_mesh.indices, &mesh.indices[0], nb_bytes_indices,
          cudaMemcpyHostToDevice);
        // Copies `nb_indices' to the Mesh struct
        cudaMemcpy(&tmp_mesh.nb_indices, &nb_indices, sizeof(size_t),
          cudaMemcpyHostToDevice);
        // Copies `material_ids'
        cudaMemcpy(&tmp_mesh.material_ids, &mesh.material_ids[0],
          nb_bytes_materials, cudaMemcpyHostToDevice);
      }
      
      cudaMalloc(d_meshes, nb_meshes * sizeof(Material));
      cudaThrowError();
      cudaMemcpy(*d_meshes, tmp_meshes, nb_meshes * sizeof (Mesh),
        cudaMemcpyHostToDevice);

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
    upload_vertices_normals(_attrib, &_d_vertices, &_d_normals);
    upload_materials(_materials, &_d_materials);
    upload_meshes(_shapes, &_d_meshes);

    _uploaded = true;
  }

  void
  Scene::release()
  {
    if (!_uploaded)
      return;
    
    cudaFree(_d_vertices);
    cudaFree(_d_normals);

    _uploaded = false;
  }

  void
  Scene::init(const char *filepath)
  { 
    _ready = tinyobj::LoadObj(&_attrib, &_shapes,
                              &_materials, &_load_error, filepath);
  }

} // namespace scene