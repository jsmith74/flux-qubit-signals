#include "BFGS_Optimization.h"
#include "Hamiltonian.h"

#include <time.h>
#include <fstream>
#include <iostream>

int main( int argc, char *argv[] ){

    if(argc != 2){

        std::cout << "./GateOptimization [integer expansion order]" << std::endl;

        return 2;

    }

    int intParam = std::atoi(argv[1]);

    std::cout << "EXPANSION ORDER: " << intParam << std::endl;

    for(int i=0;i<1;i++){

        BFGS_Optimization optimizer(1e-9,2.0,intParam);

        clock_t t1,t2;

        t1 = clock();
        optimizer.minimize();
        t2 = clock();

        float diff = (float)t2 - (float)t1;

        //std::cout << "Runtime: " << diff/CLOCKS_PER_SEC << std::endl << std::endl;

    }

    std::cout << std::endl << std::endl;

    return 0;

}
