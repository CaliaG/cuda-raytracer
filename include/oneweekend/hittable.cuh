#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include <oneweekend/ray.cuh>

class Material;

struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  Material *mat_ptr;
};

class Hittable {
public:
  Material *mat_ptr;

public:
  __device__ virtual bool hit(const Ray &r, float d_min,
                              float d_max,
                              HitRecord &rec) const = 0;
};

#endif
