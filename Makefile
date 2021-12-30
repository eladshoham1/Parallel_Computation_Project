build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o myprog main.o cFunctions.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o myprog

run:
	mpiexec -np 4 ./myprog < input.txt

runOn2:
	mpiexec -np 2 -machinefile mf -map-by node ./myprog < input.txt