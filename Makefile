CC=gcc
CFLAGS = -fopenmp

all: freq count cpi mxv

freq: char_freq.c 
	$(CC) -o freq char_freq.c $(CFLAGS) 

count: count-sort.c 
	$(CC) -o count count-sort.c $(CFLAGS) 

cpi: cpi.c
	$(CC) -o cpi cpi.c $(CFLAGS) 

mxv: mxv.c
	$(CC) -o mxv mxv.c $(CFLAGS) 

clean:
	rm freq count