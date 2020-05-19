# include <stdlib.h>
# include <stdio.h>
# include <time.h>

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


int main ( int argc, char *argv[] ) {
	float *a, *b, *c;
	float *b_device, *c_device, *a_device;
	int N;

	if (argc != 3) {
		printf ("Usage : %s <matrix size> <threads>\n", argv[0]);
		exit(1);
	}

	N = strtol(argv[1], NULL, 10);

	/*
	Allocate the matrices.
	*/
	a = ( float *)  malloc ( N * N * sizeof ( float) );
	b = ( float * ) malloc ( N * sizeof ( float ) );
	c = ( float * ) malloc ( N * sizeof ( float ) );
	
	cudaMalloc((void **) &c_device, N * sizeof(float));
	cudaMalloc((void **) &b_device, N * sizeof(float));
	cudaMalloc((void **) &a_device, N * N * sizeof(float));

	// for (int i = 0; i < N; i++) {
	// 	a[i] = ( float * ) malloc ( N * sizeof ( float ) );
	// }

	/*
	Assign values to the B and C matrices.
	*/
	srand ( time ( NULL));

	for (int i = 0; i < N; i++ ) 
		for (int j = 0; j < N; j++ )
			*(a + j + (i * N)) = ( float ) rand() / (RAND_MAX * 2.0 - 1.0);

		for (int i = 0; i < N; i++ ) {
	    b[i] = ( float ) rand() / (RAND_MAX * 2.0 - 1.0);
	    c[i] = 0.0;
	}

	cudaMemcpy(c_device, c, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a_device, a, N * N * sizeof(float), cudaMemcpyHostToDevice);

	int threads = strtol(argv[2], NULL, 10);
	int blocks = (N + threads - 1) / threads;

	calc<<<blocks, threads>>>(c_device, b_device, a_device, N);
	// /* computation */
	// for (int i = 0; i < N; i++) {
	// 	c[i] = 0.0;
	// 	for (int j = 0; j < N; j++ )
	// 		c[i] += a[i][j] * b[j];
	// }

	cudaMemcpy(c, c_device, N * sizeof(float), cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; i++ ) {
		printf("\t %1.3f ", b[i]);
		printf("\t %1.3f \n", c[i]);
	}

	return 0;

}