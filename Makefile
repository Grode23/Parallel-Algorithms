CC=gcc
CFLAGS = -fopenmp

all: freq count

freq: char_freq.c 
	$(CC) -o freq char_freq.c $(CFLAGS) 

count: count-sort.c 
	$(CC) -o count count-sort.c $(CFLAGS) 

clean:
	rm freq count