CC = icpc
CFLAGS = -O3 -c
LFLAGS = -O3 -o GateOptimization
OMPFLAGS = -openmp
INCLUDE = -inline-forceinline -I /home/jake/Documents/EIGEN
OBJS = main.o BFGS_Optimization.o MeritFunction.o Hamiltonian.o

all: GateOptimization script

script: script.cpp
	$(CC) $(OMPFLAGS) script.cpp -o script

GateOptimization: $(OBJS)
	$(CC) $(LFLAGS) $(OBJS)

main.o: main.cpp
	$(CC) $(CFLAGS) $(INCLUDE) main.cpp

Hamiltonian.o: Hamiltonian.cpp
	$(CC) $(CFLAGS) $(INCLUDE) Hamiltonian.cpp

BFGS_Optimization.o: BFGS_Optimization.cpp
	$(CC) $(CFLAGS) $(INCLUDE) BFGS_Optimization.cpp

MeritFunction.o: MeritFunction.cpp
	$(CC) $(CFLAGS) $(INCLUDE) MeritFunction.cpp

clean:
	rm *o GateOptimization *.dat script
