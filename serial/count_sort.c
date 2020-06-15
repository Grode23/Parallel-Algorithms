#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void count_sort(int array[], int size) {
	int count;

	int* temp = malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		count = 0;
		for (int j = 0; j < size; j++) {
			if (array[j] < array[i])
				count++;
			else if (array[j] == array[i] && j < i)
				count++;
		}
		temp[count] = array[i];
	}

	memcpy(array, temp, size * sizeof(int));
	free(temp);
}

void generate_array(int size, int* array, int limit) {

	//Initializion for rand();
	srand(time(NULL));

	for (int i = 0; i < size; i++){
		*(array + i) = rand() % limit;
		printf("%d\n",*(array + i) );
	}
}

int main(int argc, char *argv[]) {

	int size = 20000;
	int array[size];

	generate_array(size, array, 100);
	printf("Generated\n");

	//Starting time of solution
	clock_t start = clock();

	count_sort(array, size);
	printf("Sorting is done\n");

	//Finishing time of solution
	clock_t finish = clock();
	double time_spent = (double)(finish - start) / CLOCKS_PER_SEC;

	for(int i = 0; i < size; i++){
		printf("%d ",array[i] );
	}
	printf("\n");

	printf("Time spent: %f\n", time_spent);

	return 0;
}
