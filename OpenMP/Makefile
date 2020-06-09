CC=gcc
CFLAGS = -fopenmp

all: freq count cpi mxv
	
freq: freq_long freq_normal

freq_long: char_freq_long_files.c 
	$(CC) -o freq_long char_freq_long_files.c $(CFLAGS) 

freq_normal: char_freq_normal.c 
	$(CC) -o freq_normal char_freq_normal.c $(CFLAGS) 

count: count_sort.c 
	$(CC) -o count count_sort.c $(CFLAGS) 

cpi: cpi.c
	$(CC) -o cpi cpi.c $(CFLAGS) 

mxv: mxv.c
	$(CC) -o mxv mxv.c $(CFLAGS) 

clean:
	rm freq_long freq_normal count cpi mxv