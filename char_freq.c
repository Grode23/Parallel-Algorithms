#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define N 128
#define base 0

int main (int argc, char *argv[]) {

	char * buffer;
	char * filename;
	size_t result;
	int freq[N];

	if (argc != 2) {
		printf ("Usage : %s <file_name>\n", argv[0]);
		return 1;
	}

	filename = argv[1];
	FILE* pFile = fopen ( filename , "rb" );
	
	if (pFile == NULL) {
		printf ("File error\n"); 
		return 2;
	}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	long file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);

	// Allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char) * file_size);
	if (buffer == NULL) {
		printf ("Memory error\n"); 
		return 3;
	}

	// copy the file into the buffer:
	result = fread (buffer, 1, file_size, pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;}

	memset(freq, 0, N * sizeof(int));

	int temp[N];
	memset(temp, 0, N *sizeof(int));

	omp_set_num_threads(8);
	#pragma omp parallel firstprivate(temp)
	{

		for (int i = omp_get_thread_num(); i < file_size; i+=8) {
			temp[buffer[i] - base]++;
		}

		for(int i = 0; i < N; i++){
			#pragma omp critical
			freq[i] += temp[i];
		}

	}
	for (int j = 0; j < N; j++) {
		printf("%c = %d\n", j + base, freq[j]);
	}


	fclose (pFile);
	free (buffer);

	return 0;
}