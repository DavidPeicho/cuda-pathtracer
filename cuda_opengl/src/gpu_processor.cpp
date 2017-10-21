#include "gpu_processor.h"

namespace processor
{
  GPUProcessor::GPUProcessor(scene::Scene& scene, int width, int height)
               : _scene(scene)
               , _width(width)
               , _height(height)
  { }

  bool
  GPUProcessor::init()
  {
    // Uploads scene for GPU usage
    _scene.upload(false);
    return _scene.ready();
  }

} // namespace processor