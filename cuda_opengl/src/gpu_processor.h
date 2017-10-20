#pragma once

class GPUProcessor
{
  public:
    GPUProcessor(int w, int h);

  public:
    void
    run();

  private:
    int _width;
    int _height;
};
