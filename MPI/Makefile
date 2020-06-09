CC=mpicc

all: freq count
	
count: count_first count_second

freq: char_freq.c 
	$(CC) -o freq char_freq.c 

count_first: count_sort_first_for.c 
	$(CC) -o count_first count_sort_first_for.c

count_second: count_sort_second_for.c 
	$(CC) -o count_second count_sort_second_for.c

clean:
	rm freq count_first count_second 