#include <stdio.h>
#include <stdlib.h>

__global__ void count_sort(int *x, int *y, int size){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){

    int count = 0;

    for(int j = 0; j < size; j++){

      if (x[j] < x[idx])
        count++;
      else if (x[j] == x[idx] && j < idx)
        count++;

    }

    y[count] = x[idx];
  }

}

int main(int argc, char *argv[]) {

  int *x_host, *y_host, *x_device, *y_device;

  if (argc != 3) {
    printf ("Usage : %s <array_size> <Threads_per_block>\n", argv[0]);
    return 1;
  }

  int size = strtol(argv[1], NULL, 10);

  int threads = strtol(argv[2], NULL, 10);
  int blocks = (size + threads - 1) / threads;

  // Allocate memory as arrays on host
  x_host = (int*) malloc(size * sizeof(int));
  y_host = (int*) malloc(size * sizeof(int));

  // Allocate memory as arrays on device
  cudaMalloc( &x_device, size * sizeof(int));
  cudaMalloc((void **) &y_device, size * sizeof(int));

  for (int i=0; i<size; i++){
    x_host[i] = size - i;
    y_host[i] = 0;
  }

  // Copy data to device
  cudaMemcpy(x_device, x_host, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y_host, size * sizeof(int), cudaMemcpyHostToDevice);

  // Do the calculations
  count_sort<<<blocks, threads>>>(x_device, y_device, size);

  // Get data from device to host
  cudaMemcpy(y_host, y_device, size * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<size; i++){
    printf("%d\n", y_host[i]);
  }

  // Free variables
  free(x_host);
  free(y_host);
  cudaFree(x_device);
  cudaFree(y_device);

  return 0;
}
