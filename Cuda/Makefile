CC=nvcc

all: freq count mxv

freq: char_freq.cu
	$(CC) -o freq char_freq.cu

count: count_sort.cu 
	$(CC) -o count count_sort.cu

mxv: mxv.cu 
	$(CC) -o mxv mxv.cu

clean:
	rm freq count mxv 