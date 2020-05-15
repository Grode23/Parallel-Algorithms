#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>

#define N 128

__global__ void calc_freq(int *freq, int file_size, char *buffer, int total_threads){
	int temp[N];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;  

	for (int i = 0; i < N; i++){
		temp[i] = 0;
	}

	for(int i = idx; i < file_size; i += total_threads) {
     	temp[buffer[i]]++;	
	}

	for(int i = 0; i < N; i++){
		atomicAdd(&freq[i], temp[i]);
	}

}

int main (int argc, char *argv[]) {
	
	int *freq_host, *freq_device, *sum_device;
	char * buffer_device;
	if (argc != 4) {
		printf ("Usage : %s <file_name> <blocks> <threads_per_block>\n", argv[0]);
		return 1;
	}
	
	char *filename = argv[1];
	FILE *pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	long file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);
	
	// allocate memory to contain the file:
	char *buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	size_t result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;} 
	
	freq_host = (int*) malloc(N * sizeof(int));
	cudaMalloc((void **) &sum_device, N * sizeof(int));
	cudaMalloc((void **) &freq_device, N * sizeof(int));
	cudaMalloc((void **) &buffer_device, file_size * sizeof(char));

	for (int i = 0; i < N; i++){
		freq_host[i]=0;
	}

	cudaMemcpy(sum_device, freq_host, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_device, buffer, file_size * sizeof(char), cudaMemcpyHostToDevice);


	int threads = strtol(argv[3], NULL, 10);
	int blocks = (strtol(argv[2], NULL, 10) + threads - 1) / threads;
	int total_threads = blocks * threads;
	
	calc_freq<<<blocks, threads>>>(sum_device, file_size, buffer_device, total_threads);
	// sum_freq<<<blocks, threads>>>(freq_device, sum_device);

	cudaMemcpy(freq_host, sum_device, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int j = 0; j < N; j++){
		printf("%c = %d\n", j, freq_host[j]);
	}	

	fclose (pFile);
	free (buffer);
	cudaFree(freq_host);
	cudaFree(freq_device);
	cudaFree(sum_device);

	return 0;
}