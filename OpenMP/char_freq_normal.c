#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <omp.h>

#define N 128
#define	NUM_OF_THREADS 8

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int i, j, freq[N];

	if (argc != 2) {
		printf ("Usage : %s <file_name>\n", argv[0]);
		return 1;
	}
	filename = argv[1];
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);
	
	//Starting time of solution
	double start = omp_get_wtime();

	// allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;} 
	
	for (j=0; j<N; j++){
		freq[j]=0;
	}

	int *indexes = malloc(file_size * sizeof(int));


	for (int i=0; i<file_size; i++){
		indexes[i] = (int)buffer[i];
	}	

	long temp[N];
	//Temporary variable instead of freq because I have a lot of threads
	memset(temp, 0, N *sizeof(long));

	omp_set_num_threads(NUM_OF_THREADS);

	#pragma omp parallel firstprivate(temp)
	{

			// Count ASCII characters
		for (int i = omp_get_thread_num(); i < file_size; i+=NUM_OF_THREADS) {
			temp[buffer[i]]++;
		}

			// Add values of private temp to shared freq
		for(int i = 0; i < N; i++){
			#pragma omp critical
			freq[i] += temp[i];
		}

	}

	for (j=0; j<N; j++){
		printf("%d = %d\n", j, freq[j]);
	}	

	//Finishing time of solution
	double finish = omp_get_wtime();

	printf("Time spent: %f\n", finish - start);

	fclose (pFile);
	free (buffer);

	return 0;
}