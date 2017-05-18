#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

int main(){

#pragma omp parallel for default(none) schedule(dynamic)
    for(int i=6;i<=100;i++){
	usleep(3000000 * omp_get_thread_num());
        for(int j=0;j<20;j++){
            std::string systemCommand;
            std::stringstream ss;
            ss << i;
            ss >> systemCommand;
            systemCommand = "./GateOptimization " + systemCommand;

            system(systemCommand.c_str());
        }

    }

    return 0;
}
