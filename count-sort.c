#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define NUM_OF_THREADS 8

void count_sort(int array[], int size) {
	int count;

	int* temp = malloc(size * sizeof(int));

	// Set number of threads
	omp_set_num_threads(NUM_OF_THREADS);


	#pragma omp parallel shared(temp) private(count)
	{
		int rank = omp_get_thread_num();
		for (int i = rank; i < size; i += NUM_OF_THREADS) {
			count = 0;
			for (int j = 0; j < size; j++) {
				if (array[j] < array[i])
					count++;
				else if (array[j] == array[i] && j < i)
					count++;
			}
			#pragma omp critical
			temp[count] = array[i];
		}
	}

	memcpy(array, temp, size * sizeof(int));
	free(temp);
}

void generate_array(int size, int* array, int limit) {

	//Initializion for rand();
	srand(23);

	for (int i = 0; i < size; i++)
		*(array + i) = rand() % limit;
}


int main(int argc, char *argv[]) {

	int size = 60000;
	int array[size];

	generate_array(size, array, 100);
	printf("Generated\n");

	//Starting time of solution
	double start = omp_get_wtime();

	count_sort(array, size);
	printf("Sorting is done\n");

	//Finishing time of solution
	double finish = omp_get_wtime();

	for(int i = 0; i < size; i++){
		printf("%d ",array[i] );
	}

	printf("Time spent: %f\n", finish - start);

	return 0;
}