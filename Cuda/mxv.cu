#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Device code
__global__ void calc(float *result, float *b, float *a, int size){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size){
		
		float temp;
		
		for (int j = 0; j < size; j++){
			temp = *(a + j + (idx * size)) * (*(b + j));
			atomicAdd(&result[idx], temp);
		}
	}
}

int main (int argc, char *argv[]) {
	float *a, *b, *c;
	float *b_device, *c_device, *a_device;
	int size;

	if (argc != 3) {
		printf ("Usage : %s <matrix size> <threads>\n", argv[0]);
		exit(1);
	}

	// Get size from agruments
	size = strtol(argv[1], NULL, 10);
	
	// Allocate the matrices
	a = (float*) malloc (size * size * sizeof(float));
	b = (float*) malloc (size * sizeof(float));
	c = (float*) malloc (size * sizeof(float));
	cudaMalloc((void **) &c_device, size * sizeof(float));
	cudaMalloc((void **) &b_device, size * sizeof(float));
	cudaMalloc((void **) &a_device, size * size * sizeof(float));

	// Assign values to the B and C matrices
	srand ( time ( NULL));

	for (int i = 0; i < size; i++ ) 
		for (int j = 0; j < size; j++ )
			*(a + j + (i * size)) = ( float ) rand() / (RAND_MAX * 2.0 - 1.0);

		for (int i = 0; i < size; i++ ) {
	    b[i] = ( float ) rand() / (RAND_MAX * 2.0 - 1.0);
	    c[i] = 0.0;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(c_device, c, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a_device, a, size * size * sizeof(float), cudaMemcpyHostToDevice);

	// User can choose only the number of threads per blocks 
	// The blocks will be calculated automatically
	int threads = strtol(argv[2], NULL, 10);
	int blocks = (size + threads - 1) / threads;

	cudaEventRecord(start);

	calc<<<blocks, threads>>>(c_device, b_device, a_device, size);
	
	cudaEventRecord(stop);

	cudaMemcpy(c, c_device, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	for (int i = 0; i < size; i++ ) {
		printf("\t %1.3f ", b[i]);
		printf("\t %1.3f \n", c[i]);
	}
	
	printf("GPU time (ms): %f\n", milliseconds);

	free(a);
	free(b);
	free(c);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);

	return 0;

}