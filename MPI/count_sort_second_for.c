#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

void count_sort(int array[], int size, int numtasks, int rank) {

	int count;
	int* temp = malloc(size * sizeof(int));
	int final_count;

	for (int i = 0; i < size; i++) {
		count = 0;
		final_count = 0;
		for (int j = rank; j < size; j+= numtasks) {
			if (array[j] < array[i] || (array[j] == array[i] && j < i))
				count++;
		}

		MPI_Reduce(&count, &final_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		temp[final_count] = array[i];
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

void display(int size, int* array){
	for(int i = 0; i < size; i++) {
		printf("%d ",*(array + i) );
	}

	printf("\n");
}

int main(int argc, char *argv[]) {

	int rank, numtasks;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size = 20000;
	int array[size];

	generate_array(size, array, 100);
	// for (int i=0; i<size; i++)
	// 	array[i] = size - i;

	//Starting time of solution
	double start = MPI_Wtime();

	count_sort(array, size, numtasks, rank);

	//Finishing time of solution
	double finish = MPI_Wtime();
	if(rank == 0) {

		printf("Array after sorting: \n");
		display(size, array);

		printf("Time spent: %f\n", finish - start);

	}

	MPI_Finalize();

	return 0;
}
