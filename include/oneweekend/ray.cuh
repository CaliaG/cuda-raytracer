#ifndef RAY_CUH
#define RAY_CUH
#include <oneweekend/vec3.cuh>

class Ray {
public:
  Vec3 dir;
  Vec3 orig;
  double tm;

public:
  __device__ Ray() {}
  __device__ Ray(const Point3 &p1, const Vec3 &d1,
                 double time = 0.0f)
      : orig(p1), dir(d1), tm(time) {}
  __device__ Point3 origin() const { return orig; }
  __device__ Vec3 direction() const { return dir; }
  __device__ double time() const { return tm; }
  __device__ Point3 at(double t) const {
    return orig + dir * t;
  }
};

#endif
