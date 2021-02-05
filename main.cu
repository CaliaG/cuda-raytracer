#include "src/camera.cuh"
#include "src/cbuffer.hpp"
#include "src/color.hpp"
#include "src/debug.hpp"
#include "src/external.hpp"
#include "src/hittables.cuh"
#include "src/kernels/makeworld.cuh"
#include "src/kernels/trace.cuh"
#include "src/material.cuh"
#include "src/ray.cuh"
#include "src/sphere.cuh"
#include "src/texture.cuh"
#include "src/vec3.cuh"

void save_to_ppm(thrust::device_ptr<Vec3> fb, int nx, int ny);
void save_to_jpg(thrust::device_ptr<Vec3> fb, int nx, int ny);

__global__ void rand_init(curandState *randState, int seed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(seed, 0, 0, randState);
  }
}

__global__ void render_init(int mx, int my, curandState *randState, int seed) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    curand_init(seed, 0, 0, randState);
  }
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= mx) || (j >= my)) {
    return;
  }
  int pixel_index = j * mx + i;
  // same seed, different index
  curand_init(seed + pixel_index, pixel_index, 0, &randState[pixel_index]);
}

void get_device_props() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cerr << "Device Number: " << i << std::endl;
    std::cerr << "Device name: " << prop.name << std::endl;
    std::cerr << "Memory Clock Rate (KHz): "
              << prop.memoryClockRate << std::endl;
    std::cerr << "Memory Bus Width (bits): "
              << prop.memoryBusWidth << std::endl;
    std::cerr << "  Peak Memory Bandwidth (GB/s): "
              << 2.0 * prop.memoryClockRate *
                     (prop.memoryBusWidth / 8) / 1.0e6
              << std::endl;
  }
}

Camera makeCam(int WIDTH, int HEIGHT) {

  // one weekend final camera specification
  // Vec3 lookfrom(13, 2, 3);
  // Vec3 lookat(0, 0, 0);
  // Vec3 wup(0, 1, 0);
  // float vfov = 20.0f;
  // float aspect_r = float(WIDTH) / float(HEIGHT);
  // float dist_to_focus = 10.0;
  //(lookfrom - lookat).length();
  // float aperture = 0.1;
  // float t0 = 0.0f, t1 = 1.0f;

  // nextweek empty cornell box specification

  Vec3 lookfrom(478, 278, -600);
  Vec3 lookat(278, 278, 0);
  Vec3 wup(0, 1, 0);
  float vfov = 40.0f;
  float aspect_r = float(WIDTH) / float(HEIGHT);
  float dist_to_focus = (lookfrom - lookat).length();
  float aperture = 0.0;
  float t0 = 0.0f, t1 = 1.0f;

  Camera cam(lookfrom, lookat, wup, vfov, aspect_r,
             aperture, dist_to_focus, t0, t1);
  return cam;
}

void make_image(thrust::device_ptr<unsigned char> &imdata,
                thrust::device_ptr<int> &imwidths,
                thrust::device_ptr<int> &imhs,
                thrust::device_ptr<int> &imch) {
  std::vector<const char *> impaths = {"media/earthmap.png",
                                       "media/lsjimg.png"};
  std::vector<int> ws, hes, nbChannels;
  int totalSize;
  std::vector<unsigned char> imdata_h;
  imread(impaths, ws, hes, nbChannels, imdata_h, totalSize);
  ////// thrust::device_ptr<unsigned char> imda =
  //////    thrust::device_malloc<unsigned char>(imd.size);
  unsigned char *h_ptr = imdata_h.data();

  // --------------------- image ------------------------
  upload_to_device(imdata, h_ptr, imdata_h.size());

  int *ws_ptr = ws.data();

  upload_to_device(imwidths, ws_ptr, ws.size());

  int *hs_ptr = hes.data();
  upload_to_device(imhs, hs_ptr, hes.size());

  int *nb_ptr = nbChannels.data();
  upload_to_device(imch, nb_ptr, nbChannels.size());
}

void make_final_world(
    thrust::device_ptr<Hittable *> &hs,
    thrust::device_ptr<Hittables *> &world) {
  world = thrust::device_malloc<Hittables *>(1);
  CUDA_CONTROL(cudaGetLastError());
  // CUDA_CONTROL(upload(veri));
  int box_size = 6;
  int side_box_nb = 20;
  int sphere_nb = 10;
  int nb_hittable = side_box_nb;
  nb_hittable *= side_box_nb;
  nb_hittable *= box_size;
  nb_hittable += sphere_nb;
  // nb_hittable += 1;
  hs = thrust::device_malloc<Hittable *>(nb_hittable);
}

void make_cornell(thrust::device_ptr<Hittable *> &hs,
                  thrust::device_ptr<Hittables *> &world) {
  world = thrust::device_malloc<Hittables *>(1);
  CUDA_CONTROL(cudaGetLastError());
  // CUDA_CONTROL(upload(veri));
  int box_nb = 3;
  int box_size = 6;
  int nb_hittable = box_nb * box_size + 3;
  // nb_hittable += 1;
  hs = thrust::device_malloc<Hittable *>(nb_hittable);
}

int main() {
  float aspect_ratio = 16.0f / 9.0f;
  int WIDTH = 480;
  int HEIGHT = static_cast<int>(WIDTH / aspect_ratio);
  int BLOCK_WIDTH = 32;
  int BLOCK_HEIGHT = 18;
  int SAMPLE_NB = 40;
  int BOUNCE_NB = 20;

  get_device_props();

  std::cerr << "Resim boyutumuz " << WIDTH << "x" << HEIGHT
            << std::endl;

  std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT
            << " bloklar halinde" << std::endl;

  // declare frame size
  int total_pixel_size = WIDTH * HEIGHT;
  size_t frameSize = 3 * total_pixel_size;

  // declare frame
  thrust::device_ptr<Vec3> fb = thrust::device_malloc<Vec3>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

  // declare random state
  int SEED = time(NULL);
  thrust::device_ptr<curandState> randState1 =
      thrust::device_malloc<curandState>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

  // declare random state 2
  thrust::device_ptr<curandState> randState2 =
      thrust::device_malloc<curandState>(1);
  CUDA_CONTROL(cudaGetLastError());
  rand_init<<<1, 1>>>(thrust::raw_pointer_cast(randState2),
                      SEED);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  // declare world
  thrust::device_ptr<Hittables *> world =
      thrust::device_malloc<Hittables *>(1);
  thrust::device_ptr<Hittable *> hs;
  make_final_world(hs, world);
  // make_cornell(hs, world);

  CUDA_CONTROL(cudaGetLastError());

  // declara imdata

  // --------------------- image ------------------------
  thrust::device_ptr<unsigned char> imdata;
  thrust::device_ptr<int> imwidths;
  thrust::device_ptr<int> imhs;
  thrust::device_ptr<int> imch; // nb channels
  make_image(imdata, imwidths, imhs, imch);

  CUDA_CONTROL(cudaGetLastError());
  // make_empty_cornell_box<<<1, 1>>>(
  //    thrust::raw_pointer_cast(world),
  //    thrust::raw_pointer_cast(hs),
  //    thrust::raw_pointer_cast(randState2));

  make_world<<<1, 1>>>(thrust::raw_pointer_cast(world),
                       thrust::raw_pointer_cast(hs),
                       thrust::raw_pointer_cast(randState2),
                       20, thrust::raw_pointer_cast(imdata),
                       thrust::raw_pointer_cast(imwidths),
                       thrust::raw_pointer_cast(imhs),
                       thrust::raw_pointer_cast(imch));
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  clock_t baslar, biter;
  baslar = clock();

  dim3 blocks(WIDTH / BLOCK_WIDTH + 1,
              HEIGHT / BLOCK_HEIGHT + 1);
  dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
  render_init<<<blocks, threads>>>(
      WIDTH, HEIGHT, thrust::raw_pointer_cast(randState1),
      SEED + 7);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  // declare camera
  Camera cam = makeCam(WIDTH, HEIGHT);

  render<<<blocks, threads>>>(
      thrust::raw_pointer_cast(fb), WIDTH, HEIGHT,
      SAMPLE_NB, BOUNCE_NB, cam,
      thrust::raw_pointer_cast(world),
      thrust::raw_pointer_cast(randState1));
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());
  biter = clock();
  double saniyeler =
      ((double)(biter - baslar)) / CLOCKS_PER_SEC;
  std::cerr << "Islem " << saniyeler << " saniye surdu" << std::endl;

  // save_to_ppm(fb, WIDTH, HEIGHT);
  save_to_jpg(fb, WIDTH, HEIGHT);
  
  CUDA_CONTROL(cudaDeviceSynchronize());
  CUDA_CONTROL(cudaGetLastError());
  free_world(fb,                           //
             world,                        //
             hs,                           //
             imdata, imch, imhs, imwidths, //
             randState1,                   //
             randState2);
  // free_world(fb, world, hs, randState1, randState2);
  // free_empty_cornell(fb, world, hs, randState1,
  // randState2);
  CUDA_CONTROL(cudaGetLastError());

  cudaDeviceReset();
}

void save_to_ppm(thrust::device_ptr<Vec3> fb, int nx, int ny) {
	std::ofstream ofs;
	ofs.open("image.ppm", std::ios::out | std::ios::binary);
	ofs << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            thrust::device_reference<Vec3> pix_ref = fb[pixel_index];
            Vec3 pixel = pix_ref;
            int ir = int(255.99*pixel.r());
            int ig = int(255.99*pixel.g());
            int ib = int(255.99*pixel.b());
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
	ofs.close();
}

void save_to_jpg(thrust::device_ptr<Vec3> fb, int nx, int ny) {
    uint8_t* imgBuff = (uint8_t*)std::malloc(nx * ny * 3 * sizeof(uint8_t));
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t index = j*nx + i;
            thrust::device_reference<Vec3> pix_ref = fb[index];
            Vec3 pixel = pix_ref;
            float r = pixel.r();
            float g = pixel.g();
            float b = pixel.b();
            // stbi generates a Y flipped image
            size_t rev_index = (ny - j - 1) * nx + i;
            imgBuff[rev_index * 3 + 0] = int(255.999f * r) & 255;
            imgBuff[rev_index * 3 + 1] = int(255.999f * g) & 255;
            imgBuff[rev_index * 3 + 2] = int(255.999f * b) & 255;
        }
    }
    std::cout << "FUCK\n";
    //stbi_write_png("out.png", WIDTH, HEIGHT, 3, imgBuff, WIDTH * 3);
    stbi_write_jpg("image.jpg", nx, ny, 3, imgBuff, 100);
    std::free(imgBuff);
}