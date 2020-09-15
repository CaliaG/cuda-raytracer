// libs
#include <nextweek/camera.cuh>
#include <nextweek/cbuffer.hpp>
#include <nextweek/color.hpp>
#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>
#include <nextweek/hittables.cuh>
#include <nextweek/kernels/makeworld.cuh>
#include <nextweek/kernels/trace.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/texture.cuh>
#include <nextweek/vec3.cuh>

__global__ void rand_init(curandState *randState,
                          int seed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(seed, 0, 0, randState);
  }
}

__global__ void render_init(int mx, int my,
                            curandState *randState,
                            int seed) {
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
  curand_init(seed, pixel_index, 0,
              &randState[pixel_index]);
}

__global__ void free_world(Hittables **world,
                           Hittable **ss) {
  int size = 22 * 22 + 1 + 3;
  for (int i = 0; i < size; i++) {
    delete ((Hittable *)ss[i])->mat_ptr;
    delete ss[i];
  }
  delete world[0];
}

void freeEverything(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<Hittables *> &world,
    thrust::device_ptr<Hittable *> &hs,
    thrust::device_ptr<unsigned char> imdata,
    thrust::device_ptr<int> imch,
    thrust::device_ptr<int> imhs,
    thrust::device_ptr<int>(imwidths),
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {
  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(world);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(hs);
  CUDA_CONTROL(cudaGetLastError());
  // dcam.free();
  thrust::device_free(imdata);
  thrust::device_free(imch);
  thrust::device_free(imhs);
  thrust::device_free(imwidths);
  // free(ws_ptr);
  // free(nb_ptr);
  // free(hs_ptr);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState2);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState1);
  CUDA_CONTROL(cudaGetLastError());
}

int main() {
  float aspect_ratio = 16.0f / 9.0f;
  int WIDTH = 320;
  int HEIGHT = static_cast<int>(WIDTH / aspect_ratio);
  int BLOCK_WIDTH = 10;
  int BLOCK_HEIGHT = 10;
  int SAMPLE_NB = 30;
  int BOUNCE_NB = 20;

  std::cerr << "Resim boyutumuz " << WIDTH << "x" << HEIGHT
            << std::endl;

  std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT
            << " bloklar halinde" << std::endl;

  // declare frame size
  int total_pixel_size = WIDTH * HEIGHT;
  size_t frameSize = 3 * total_pixel_size;

  // declare frame
  thrust::device_ptr<Vec3> fb =
      thrust::device_malloc<Vec3>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

  // declare random state
  int SEED = 1987;
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
  CUDA_CONTROL(cudaGetLastError());
  int row = 22;
  int focus_obj_nb = 3;
  int nb_hittable = row * row + 1 + focus_obj_nb;
  thrust::device_ptr<Hittable *> hs =
      thrust::device_malloc<Hittable *>(nb_hittable);
  CUDA_CONTROL(cudaGetLastError());

  // declara imdata
  std::vector<const char *> impaths = {"media/earthmap.png",
                                       "media/lsjimg.png"};
  std::vector<int> ws, hes, nbChannels;
  int totalSize;
  std::vector<unsigned char> imdata_h;
  imread(impaths, ws, hes, nbChannels, imdata_h, totalSize);
  // thrust::device_ptr<unsigned char> imda =
  //    thrust::device_malloc<unsigned char>(imd.size);
  unsigned char *h_ptr = imdata_h.data();

  // --------------------- image ------------------------
  thrust::device_ptr<unsigned char> imdata;
  upload_to_device(imdata, h_ptr, imdata_h.size());
  // CUDA_CONTROL(cudaMalloc(&imdata, sizeof(unsigned char)
  // *
  //                                     totalSize));
  // CUDA_CONTROL(cudaMemcpy((void *)imdata,
  //                        (const void *)h_ptr,
  //                        totalSize * sizeof(unsigned
  //                        char),
  //                        cudaMemcpyHostToDevice));

  int *ws_ptr = ws.data();

  thrust::device_ptr<int> imwidths;
  upload_to_device(imwidths, ws_ptr, ws.size());

  thrust::device_ptr<int> imhs;
  int *hs_ptr = hes.data();
  upload_to_device(imhs, hs_ptr, hes.size());

  thrust::device_ptr<int> imch; // nb channels
  int *nb_ptr = nbChannels.data();
  upload_to_device(imch, nb_ptr, nbChannels.size());

  CUDA_CONTROL(cudaGetLastError());

  make_world<<<1, 1>>>(
      thrust::raw_pointer_cast(world),
      thrust::raw_pointer_cast(hs), WIDTH, HEIGHT,
      thrust::raw_pointer_cast(randState2), row,
      thrust::raw_pointer_cast(imdata),
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
  Vec3 lookfrom(13, 2, 3);
  Vec3 lookat(0, 0, 0);
  Vec3 wup(0, 1, 0);
  float vfov = 20.0f;
  float aspect_r = float(WIDTH) / float(HEIGHT);
  float dist_to_focus = 10.0;
  (lookfrom - lookat).length();
  float aperture = 0.1;
  float t0 = 0.0f, t1 = 1.0f;
  Camera cam(lookfrom, lookat, wup, vfov, aspect_r,
             aperture, dist_to_focus, t0, t1);

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
  std::cerr << "Islem " << saniyeler << " saniye surdu"
            << std::endl;

  std::cout << "P3" << std::endl;
  std::cout << WIDTH << " " << HEIGHT << std::endl;
  std::cout << "255" << std::endl;

  for (int j = HEIGHT - 1; j >= 0; j--) {
    for (int i = 0; i < WIDTH; i++) {
      size_t pixel_index = j * WIDTH + i;
      thrust::device_reference<Vec3> pix_ref =
          fb[pixel_index];
      Vec3 pixel = pix_ref;
      int ir = int(255.99 * pixel.r());
      int ig = int(255.99 * pixel.g());
      int ib = int(255.99 * pixel.b());
      std::cout << ir << " " << ig << " " << ib
                << std::endl;
    }
  }
  CUDA_CONTROL(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(thrust::raw_pointer_cast(world),
                       thrust::raw_pointer_cast(hs));
  CUDA_CONTROL(cudaGetLastError());
  freeEverything(fb, world, hs, imdata, imch, imhs,
                 imwidths, randState1, randState2);
  CUDA_CONTROL(cudaGetLastError());

  cudaDeviceReset();
}
