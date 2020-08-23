#ifndef RAY_CUH
#define RAY_CUH
#include <oneweekend/vec3.cuh>

class Ray {
public:
  Vec3 dir;
  Vec3 orig;

public:
  __device__ Ray() {}
  __device__ Ray(const Point3 &p1, const Vec3 &d1) : orig(p1), dir(d1) {}
  __device__ Point3 origin() const { return orig; }
  __device__ Vec3 direction() const { return dir; }
  __device__ Point3 at(float t) const { return orig + dir * t; }
};

#endif
