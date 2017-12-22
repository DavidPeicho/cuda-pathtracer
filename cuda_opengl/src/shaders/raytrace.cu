#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <math.h>
#include <stdbool.h>

#include <driver/cuda_helper.h>
#include <math_functions.h>
#include <scene/scene_data.h>
#include <shaders/brdf.cuh>
#include <shaders/intersection.cuh>
#include <shaders/post_process.cuh>
#include <shaders/volume.cuh>
#include <utils/utils.h>

using post_process_t = float3 (*)(const float3&);

post_process_t h_post_process_table[4];

surface<void, cudaSurfaceType2D> surf;
texture<float4, cudaTextureTypeCubemap> cubemap_ref;

union rgba_24
{
  uint1 b32;

  struct
  {
    unsigned r : 8;
    unsigned g : 8;
    unsigned b : 8;
    unsigned a : 8;
  };
};

#ifndef M_PI
#define M_PI 3.14159265359f
#endif

__device__ inline float3
radiance(scene::Ray& r, const struct scene::Scenes& scenes,
         unsigned int scene_id, const scene::Camera* const cam,
         curandState* rand_state, int is_static, int static_samples)
{
  float3 acc = make_float3(0.0f);
  // For energy compensation on Russian roulette
  float3 throughput = make_float3(1.0f);

  // Contains information about each intersection.
  // This will be updated at each call to 'intersect'.
  IntersectionData inter;

  if (!is_static)
	if (intersect(r, scenes, scene_id, inter))
	    return inter.diffuse_col;
	else
	{
	    // Accumulate Environment map's contribution (approximated as many far away lights)
	    auto val = texCubemap(cubemap_ref, r.dir.x, r.dir.y, -r.dir.z);
	    return make_float3(val.x, val.y, val.z);
	}

  // Max bounces
  // Bounce more when the camera is not moving
  const int max_bounces = 1 + is_static * (static_samples + 1);
  for (int b = 0; b < max_bounces; b++) {
    float3 oriented_normal;

    float r1 = curand_uniform(rand_state);
    if (intersect(r, scenes, scene_id, inter)) {
      inter.normal = inter.normal;
      float cos_theta = dot(inter.normal, r.dir);
      oriented_normal =
        inter.normal; // cos_theta < 0 ? inter.normal : inter.normal * -1.0f;

      // return oriented_normal;

      float3 up = make_float3(0.0f, 1.0f, 0.0f);
      float3 right = cross(up, inter.normal);
      up = cross(inter.normal, right);

      // Oren-Nayar diffuse
      // BRDF = brdf_oren_nayar(cos_theta, cos_theta, light_dir, r.dir,
      // oriented_normal, 0.5f, 0.5f, inter.diffuse_col);

      // Specular ray
      // Computed everytime and then used to simulate roughness by concentrating
      // rays towards it
      float3 spec = normalize(reflect(r.dir, inter.normal));
      float PDF = pdf_lambert(); // Divided by PI
      // Lambert BRDF/PDF
      float3 BRDF = brdf_lambert(inter.diffuse_col); // Divided by PI
      float3 direct_light = BRDF / PDF;
      // Default IOR (Index Of Refraction) is 1.0f
      if (inter.ior == 1.0f || inter.light != NULL) {
        // Accumulate light emission
        if (inter.light != NULL) {
          BRDF = make_float3(inter.light->color.x, inter.light->color.y,
                             inter.light->color.z);

          acc += BRDF * inter.light->emission * throughput;
        }

        // Sample the hemisphere with a random ray
        float phi = 2.0f * M_PI *
                    curand_uniform(rand_state); // glm::mix(1.0f -
                                                // inter.specular_col,
                                                // inter.specular_col, 0.1f);

        float sin_t = __fsqrt_rn(r1);
        float cos_t = __fsqrt_rn(1.f - r1);

        // u, v and oriented_normal form the base of the hemisphere
        float3 u = normalize(cross(fabs(oriented_normal.x) > .1
                                     ? make_float3(0.0f, 1.0f, 0.0f)
                                     : make_float3(1.0f, 0.0f, 0.0f),
                                   oriented_normal));
        float3 v = cross(oriented_normal, u);

        // Diffuse hemishphere reflection
        float3 d = normalize(v * sin_t * __cosf(phi) + u * __sinf(phi) * sin_t +
                             oriented_normal * cos_t);

        r.origin += r.dir * inter.dist;

        // Mix the specular and random diffuse ray by the "specular_col" amount
        // to approximate roughness
        r.dir = mix(d, spec, inter.specular_col);

        // Avoids self intersection
        r.origin += r.dir * 0.03f;

        throughput *= direct_light;
      } else {
        // Transmision
        // n1: IOR of exterior medium
        float n1 = 1.0f; // sin theta2
        // n2: IOR of entering medium
        float n2 = inter.ior; // sin theta1
        oriented_normal = cos_theta < 0 ? inter.normal : inter.normal * -1.0f;
        float c1 = dot(oriented_normal, r.dir);
        bool entering = dot(inter.normal, oriented_normal) > 0;
        // Snell's Law
        float eta = entering ? n1 / n2 : n2 / n1;
        float eta_2 = eta * eta;

        float c2_term = 1.0f - eta_2 * (1.0f - c1 * c1);
        // Total Internal Reflection
        if (c2_term < 0.0f) {
          r.origin += oriented_normal * inter.dist / 100.f;

          // return make_float3(1.0f, 0.0f, 0.0f);
          r.dir = spec;
        } else {
          // Schlick R0
          float R0 = (n2 - n1) / (n1 + n2);
          R0 *= R0;
          float c2 = __fsqrt_rn(c2_term);
          float3 T = normalize(eta * r.dir + (eta * c1 - c2) * oriented_normal);

          float f_cos_theta = 1.0f - (entering ? -c1 : dot(T, inter.normal));
          f_cos_theta = powf(cos_theta, 5.0f);
          // Fresnel-Schlick approximation for the reflection amount
          float f_r = R0 + (1.0f - R0) * f_cos_theta;

          // If reflection
          // Not exactly sure why "0.25f" works better than "f_r"...
          if (curand_uniform(rand_state) < 0.25f) {
            throughput *= f_r * direct_light;

            r.origin += oriented_normal * inter.dist / 100.f;

            r.dir = spec; // mix(d, spec, inter.specular_col);

            // return make_float3(0.0f, 1.0f, 0.0f);
          } else // Transmission
          {
            // Energy conservation
            float f_t = 1.0f - f_r;

            throughput *= f_t * direct_light;

            // We're inside a mesh doing transmission, so we try to reduce the
            // bias as much as possible
            // or the ray could get outside of the mesh which makes no sense
            r.origin += oriented_normal * inter.dist / 10000.f;

            r.dir = T;
            // return make_float3(0.0f, 0.0f, 1.0f);
          }
        }
      }
    } else {
      // away lights)
      auto val = texCubemap(cubemap_ref, r.dir.x, r.dir.y, -r.dir.z);
      acc += make_float3(val.x, val.y, val.z) * throughput;
    }

    // Russian roulette for early path termination
    float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
    if (r1 > p && b > 1)
      return acc;

    throughput *= __fdividef(1.0f, p);
  }

  return acc;
}


__global__ void
kernel(const unsigned int width, const unsigned int height,
       const scene::Scenes scenes, unsigned int scene_id, scene::Camera cam,
       unsigned int hash_seed, int frame_nb, float3* temporal_framebuffer,
       bool moved, post_process_t post)
{
  const unsigned int half_w = width / 2;
  const unsigned int half_h = height / 2;

  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const unsigned int tid =
    (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;

  union rgba_24 rgbx;
  rgbx.a = 0.0f;

  curandState rand_state;
  curand_init(hash_seed + tid, 0, 0, &rand_state);

  scene::Ray r = generateRay(x, y, half_w, half_h, cam);

  // Depth-Of-Field
  camera_dof(r, cam, &rand_state);

  int is_static = !moved;
  int static_samples = 1;

  float3 rad = make_float3(0.0f);
    //radiance(r, scenes, scene_id, &cam, &rand_state, is_static, static_samples);

  rad = clamp(rad, 0.0f, 1.0f);
  
  float2 xy = make_float2(x, height - y);
  float2 res = make_float2(width, height);
  float2 uv = xy / res;
  float2 uv2 = 2.0f* xy/ res - 1.0f;
  float3 albedo = make_float3(0.0f, 0.0f, 0.0f);
  float3 normal = make_float3(0.0f, 0.0f, 0.0f);
  float4 scatTrans = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  float3 camPos = albedo;// make_float3(20.0f, 18.0f, -60.0f);
  float3 camX = make_float3(1.0f, 0.0f, 0.0f) *0.75;
  float3 camY = make_float3(0.0f, 1.0f, 0.0f) *0.5;
  float3 camZ = make_float3(0.0f, 0.0f, 1.0f);
  camPos += cam.position * 50.f;
  volume_raymarch(r, albedo, scatTrans);

  //lighting
  float3 color = (albedo / 3.14);// * evaluateLight(finalPos, normal) * volumetricShadow(finalPos, LPOS);
  // Apply scattering/transmittance
  color = color * scatTrans.w + make_float3(scatTrans.x, scatTrans.y, scatTrans.z);
  color = clamp(color, 0.0f, 1.0f);

  // Accumulation buffer for when the camera is static
  // This makes the image converge
  int i = (height - y - 1) * width + x;

  // Zero-out if the camera is moving to reset the buffer
  temporal_framebuffer[i] *= is_static;
  temporal_framebuffer[i] += color;

  rad = color;// temporal_framebuffer[i] / (float)frame_nb;

  // Tone Mapping + White Balance
  rad = exposure(rad);
  // Gamma Correction
  rad = pow(rad, 1.0f / 2.2f);
  rad = (*post)(rad);

  rgbx.r = rad.x * 255;
  rgbx.g = rad.y * 255;
  rgbx.b = rad.z * 255;

  surf2Dwrite(rgbx.b32, surf, x * sizeof(rgbx), y, cudaBoundaryModeZero);
}

// Very nice and fast PRNG
// Credit: Thomas Wang
inline unsigned int
WangHash(unsigned int a)
{
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);

  return a;
}

cudaError_t
raytrace(cudaArray_const_t array, const scene::Scenes& scenes,
         unsigned int scene_id, const std::vector<scene::Cubemap>& cubemaps,
         int cubemap_id, const scene::Camera* const cam,
         const unsigned int width, const unsigned int height,
         cudaStream_t stream, float3* temporal_framebuffer, bool moved,
         unsigned int post_id)
{
  // Seed for the Wang Hash
  static unsigned int seed = 0;

  if (moved)
    seed = 0;
  seed++;

  cudaBindSurfaceToArray(surf, array);

  const scene::Cubemap& cubemap = cubemaps[cubemap_id];
  cubemap_ref.addressMode[0] = cudaAddressModeWrap;
  cubemap_ref.addressMode[1] = cudaAddressModeWrap;
  cubemap_ref.filterMode = cudaFilterModeLinear;
  cubemap_ref.normalized = true;
  cudaBindTextureToArray(cubemap_ref, cubemap.cubemap, cubemap.cubemap_desc);

  // Register occupancy : nb_threads = regs_per_block / 32
  // Shared memory occupancy : nb_threads = shared_mem / 32
  // Block size occupancy

  dim3 threads_per_block(16, 16);
  dim3 nb_blocks(width / threads_per_block.x + 1,
                 height / threads_per_block.y + 1);

  if (nb_blocks.x > 0 && nb_blocks.y > 0)
    kernel<<<nb_blocks, threads_per_block, 0, stream>>>(
      width, height, scenes, scene_id, *cam, WangHash(seed), seed,
      temporal_framebuffer, moved, h_post_process_table[post_id]);

  return cudaSuccess;
}

__device__ float3
no_post_process(const float3& color)
{
  return color;
}

__device__ float3
grayscale(const float3& color)
{
  const float gray = color.x * 0.3 + color.y * 0.59 + color.z * 0.11;
  return make_float3(gray, gray, gray);
}

__device__ float3
sepia(const float3& color)
{
  return make_float3(color.x * 0.393 + color.y * 0.769 + color.z * 0.189,
                     color.x * 0.349 + color.y * 0.686 + color.z * 0.168,
                     color.x * 0.272 + color.y * 0.534 + color.z * 0.131);
}

__device__ float3
invert(const float3& color)
{
  return make_float3(1.0f - color.x, 1.0f - color.y, 1.0f - color.z);
}

__device__ post_process_t p_none = no_post_process;
__device__ post_process_t p_gray = grayscale;
__device__ post_process_t p_sepia = sepia;
__device__ post_process_t p_invert = invert;

// Copy the pointers from the function tables to the host side.
void
setupFunctionTables()
{
  cudaMemcpyFromSymbol(&h_post_process_table[0], p_none,
                       sizeof(post_process_t));
  cudaCheckError();
  cudaMemcpyFromSymbol(&h_post_process_table[1], p_gray,
                       sizeof(post_process_t));
  cudaCheckError();
  cudaMemcpyFromSymbol(&h_post_process_table[2], p_sepia,
                       sizeof(post_process_t));
  cudaCheckError();
  cudaMemcpyFromSymbol(&h_post_process_table[3], p_invert,
                       sizeof(post_process_t));
  cudaCheckError();
}
