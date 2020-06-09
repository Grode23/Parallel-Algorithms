#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <limits.h>

#define N 128
#define SIZE_OF_BUFFER 100000
#define	NUM_OF_THREADS 8

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

	//Starting time of solution
	double start = omp_get_wtime();

	//Flag is true
	// When flag becomes false, stop executing the while loop
	// And start the sum  
	int flag = 1;

	//Initialize freq outside of the loop because I am going to use it in every repetition
	memset(freq, 0, N * sizeof(int));

	while(flag){
		// For loop is going to execute 'limit' times
		int limit = file_size;
		
		if(file_size > SIZE_OF_BUFFER){
			
			// Get the first size_of_buffer letters
			buffer = (char*) malloc (sizeof(char) * SIZE_OF_BUFFER);
			
			// Reduce file_size for the next repetition
			file_size -=SIZE_OF_BUFFER;

			limit = SIZE_OF_BUFFER;
		
			// copy the file into the buffer:
			result = fread (buffer, 1, SIZE_OF_BUFFER, pFile);
			
			if (result != SIZE_OF_BUFFER) {
				printf ("Reading error\n"); 
				return 4;		
			}
	
		} else {

			//While loop becomes false
			flag = 0;

			// Take the rest of the file
			buffer = (char*) malloc (sizeof(char) * file_size);
			
			// copy the file into the buffer:
			result = fread (buffer, 1, file_size, pFile);
			
			if (result != file_size) {
				printf ("Reading error\n"); 
				return 4;
			}
		}
		// Error handling for malloc in buffer
		if (buffer == NULL || buffer==0x0) {
			printf ("Memory error\n"); 
			return 3;
		}

		// Numbers might be enormous, so I need long variable
		long temp[N];

		//Temporary variable instead of freq because I have a lot of threads
		memset(temp, 0, N *sizeof(long));

		omp_set_num_threads(NUM_OF_THREADS);

		#pragma omp parallel firstprivate(temp)
		{

			// Count ASCII characters
			for (int i = omp_get_thread_num(); i < limit; i+=NUM_OF_THREADS) {
				temp[buffer[i]]++;
			}

			// Add values of private temp to shared freq
			for(int i = 0; i < N; i++){
				#pragma omp critical
				freq[i] += temp[i];
			}

		}
	}

	long total = 0;

	for (int j = 0; j < N; j++) {
		printf("%d = %d\n", j, freq[j]);
		total += freq[j];
	}
	printf("%ld\n",total );

	//Finishing time of solution
	double finish = omp_get_wtime();

	printf("Time spent: %f\n", finish - start);

	fclose (pFile);
	free (buffer);

	return 0;
}