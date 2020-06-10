#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;
    double *local_sum;
    int i;
   
    if (argc != 2) {
		printf ("Usage : %s <number_of_intervals>\n", argv[0]);
		return 1;
    }

    //Starting time of solution
    double start = omp_get_wtime();

    long int n = strtol(argv[1], NULL, 10);
    pi = 0.0;
    h  = 1.0 / (double) n;
    sum = 0.0;

    #pragma omp parallel for reduction(+:sum) private(x)
    for (i = 1; i <= n; i++)
    {
    	x = h * ((double)i - 0.5);
        sum += 4.0 / (1.0 + x*x);
    }

    pi = h * sum;
    printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
    
    //Finishing time of solution
    double finish = omp_get_wtime();
    printf("Time spent: %f\n", finish - start);

   return 0;
}

            