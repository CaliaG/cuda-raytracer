#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <nextweek/hittable.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/texture.cuh>
#include <nextweek/utils.cuh>

struct HitRecord;

__host__ __device__ float fresnelCT(float costheta,
                                    float ridx) {
  // cook torrence fresnel equation
  float etao = 1 + sqrt(ridx);
  float etau = 1 - sqrt(ridx);
  float eta = etao / etau;
  float g = sqrt(pow(eta, 2) + pow(costheta, 2) - 1);
  float g_c = g - costheta;
  float gplusc = g + costheta;
  float gplus_cc = (gplusc * costheta) - 1;
  float g_cc = (g_c * costheta) + 1;
  float oneplus_gcc = 1 + pow(gplus_cc / g_cc, 2);
  float half_plus_minus = 0.5 * pow(g_c / gplusc, 2);
  return half_plus_minus * oneplus_gcc;
}

__host__ __device__ bool refract(const Vec3 &v,
                                 const Vec3 &n,
                                 float ni_over_nt,
                                 Vec3 &refracted) {
  Vec3 uv = to_unit(v);
  float dt = dot(uv, n);
  float discriminant =
      1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted =
        ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else {
    return false;
  }
}

__host__ __device__ Vec3 reflect(const Vec3 &v,
                                 const Vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}
class Material {
public:
  __host__ __device__ virtual ~Material() {}
  __device__ virtual bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const = 0;

  __host__ __device__ virtual Color
  emitted(float u, float v, const Point3 &p) const {
    //
    return Color(0.0f);
  }
};

class Lambertian : public Material {
public:
  __host__ __device__ Lambertian() {}
  __host__ __device__ Lambertian(const Vec3 &a) {
    albedo = new SolidColor(a);
  }
  __host__ __device__ Lambertian(Texture *a) { albedo = a; }

  __host__ __device__ ~Lambertian() { delete albedo; }
  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Color &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {
    Vec3 target = rec.normal +
                  random_in_unit_sphere(local_rand_state);
    scattered = Ray(rec.p, target, r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

  Texture *albedo;
};

class Metal : public Material {
public:
  __host__ __device__ Metal() {}
  __host__ __device__ Metal(const Color &a, float f) {
    if (f < 1)
      fuzz = f;
    else
      fuzz = 1;
    //
    albedo = new SolidColor(a);
  }
  __host__ __device__ Metal(Texture *txt, float f)
      : albedo(txt) {
    if (f < 1)
      fuzz = f;
    else
      fuzz = 1;
  }
  __host__ __device__ ~Metal() { delete albedo; }

  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {

    Vec3 reflected =
        reflect(to_unit(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected +
                               fuzz * random_in_unit_sphere(
                                          local_rand_state),
                    r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }
  Texture *albedo;
  float fuzz;
};

class Dielectric : public Material {
public:
  __host__ __device__ Dielectric() {}
  __host__ __device__ Dielectric(float ri) : ref_idx(ri) {}
  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {
    Vec3 outward_normal;
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = Vec3(1.0);
    Vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx;
      cosine = dot(r_in.direction(), rec.normal) /
               r_in.direction().length();
      cosine = sqrt(
          1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    } else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / ref_idx;
      cosine = -dot(r_in.direction(), rec.normal) /
               r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal,
                ni_over_nt, refracted))
      reflect_prob = fresnelCT(cosine, ref_idx);
    else
      reflect_prob = 1.0f;
    unsigned int seed = curand(local_rand_state);
    if (randf(seed) < reflect_prob)
      scattered = Ray(rec.p, reflected, r_in.time());
    else
      scattered = Ray(rec.p, refracted, r_in.time());
    return true;
  }
  float ref_idx;
};

class DiffuseLight : public Material {
public:
  Texture *albedo;

public:
  __host__ __device__ DiffuseLight() {}
  __host__ __device__ DiffuseLight(Texture *t) : albedo(t) {}
  __host__ __device__ DiffuseLight(Color c)
      : albedo(new SolidColor(c)) {}
  __host__ __device__ ~DiffuseLight() { delete albedo; }

  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {
    return false;
  }
  __host__ __device__ Color emitted(
      float u, float v, const Point3 &p) const override {
    return albedo->value(u, v, p);
  }
};

class Isotropic : public Material {
public:
  __host__ __device__ Isotropic() {}
  __host__ __device__ Isotropic(Color c)
      : albedo(new SolidColor(c)) {}
  __host__ __device__ Isotropic(Texture *a) : albedo(a) {}

  __device__ bool scatter(const Ray &r_in,
                          const HitRecord &rec,
                          Color &attenuation,
                          Ray &scattered,
                          curandState *loc) const override {
    scattered =
        Ray(rec.p, random_in_unit_sphere(loc), r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

public:
  Texture *albedo;
};

#endif
