#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

#define N 128

int main (int argc, char *argv[]) {

	long file_size;
	size_t result;
	int freq[N];

	int rank, numtasks;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc != 2) {
		printf ("Usage : %s <file_name>\n", argv[0]);
		return 1;
	}
	char* filename = argv[1];
	FILE* pFile = fopen ( filename, "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size
	fseek(pFile, 0, SEEK_END);
	file_size = ftell(pFile);
	rewind(pFile);
	if(rank == 0) {
		printf("file size is %ld\n", file_size);
	}

	//Starting time of solution
	double start = MPI_Wtime();

	// allocate memory to contain the file:
	char* buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;}

	for (int j=0; j<N; j++) {
		freq[j]=0;
	}

	for (int i=rank; i<file_size; i+=numtasks) {
		freq[buffer[i]]++;
	}

	int final[N];

	MPI_Reduce (&freq,&final,N, MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	if(rank == 0) {
		long total = 0;

		for (int j = 0; j < N; j++) {
			printf("%d = %d\n", j, final[j]);
			total += freq[j];
		}
		printf("Total amount of characters: %ld\n",total );
		
		//Finishing time of solution
		double finish = MPI_Wtime();

		printf("Time spent: %f\n", finish - start);

	}

	fclose (pFile);
	free(buffer);

	MPI_Finalize();

	return 0;
}
