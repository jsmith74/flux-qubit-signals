#include "MeritFunction.h"
#define SPARSETHRESHOLD 1e-30

/** ========== SET SOME STUFF ================================================== */

//#define USE_DENSE_STATES

#define USE_SPARSE_STATES

//#define USE_MATRIX_EXP_EVOLUTION   //NOTE MATRIX_EXP_EVOLUTION ONLY WORKS WITH ONE CONTROL SIGNAL

#define USE_RK4

/** ============================================================================ */


void MeritFunction::setMeritFunction(int intParam){

    int QDimension = 3;
    int numbQubits = 4;
    int constVariables = 2;
    numbControlSignals = 1;
    expansionOrder = intParam;

    tInit = 0.0;
    tFin = 0.075;
    dt = 1e-5;

    H.setHamiltonian(QDimension,numbQubits,constVariables,numbControlSignals);

    setInitialStates();

    setIdealInverseOp();

    setProjectors();

    constantVariables = constVariables;

    return;

}

Eigen::VectorXd MeritFunction::setInitialPosition(){

    funcDimension = numbControlSignals * expansionOrder + constantVariables;

    Eigen::VectorXd position(funcDimension);

    H.setTimeGrid(tInit,tFin,dt,expansionOrder);

    H.setInitialConstants(position);

    H.setInitialControlAmplitudes(position);

    H.setBaseHamiltonian(position);

    initTimeStep = 0;
    finTimeStep = H.timeStepsEvol;

    #ifdef USE_MATRIX_EXP_EVOLUTION

        U.resize(H.timeStepsEvol);
        Ugrad.resize(H.timeStepsEvol);
        UProd.resize(H.timeStepsEvol);

        H.setGradientGrid(gradientGrid);

    #endif // USE_MATRIX_EXP_EVOLUTION

    return position;

}

double MeritFunction::f(Eigen::VectorXd& position){

    double output = 0.0;

    #ifdef USE_MATRIX_EXP_EVOLUTION

        #ifdef USE_DENSE_STATES

            states = UProd[H.timeStepsEvol - 1] * statesInitial;

            states = idealInverseOp * states;


            for(int i=0;i<36;i++){

                output += (states.col(i).conjugate().transpose() * proj[i] * states.col(i)).norm();

                //std::cout << (states.col(i).conjugate().transpose() * proj[i] * states.col(i)).norm() << std::endl;

            }

            //assert(1>2 && "matrixExp check");

        #endif // USE_DENSE_STATES

        #ifdef USE_SPARSE_STATES

            assert(1>2 && "TO DO: WRITE THIS CODE");

        #endif // USE_SPARSE_STATES


    #endif // USE_MATRIX_EXP_EVOLUTION

    #ifdef USE_RK4

        #ifdef USE_DENSE_STATES

            states = statesInitial;

            H.evolveStates(initTimeStep,finTimeStep,states,position);

            states = idealInverseOp * states;

            for(int i=0;i<36;i++){

                output += (states.col(i).conjugate().transpose() * proj[i] * states.col(i)).norm();

                //std::cout << (states.col(i).conjugate().transpose() * proj[i] * states.col(i)).norm() << std::endl;

            }

            //assert(1>2 && "rk4 check");

        #endif // USE_DENSE_STATES

        #ifdef USE_SPARSE_STATES

            statesSparse = statesInitialSparse;

            H.evolveStatesSparse(initTimeStep,finTimeStep,statesSparse,position);

            statesSparse = idealInverseOpSparse * statesSparse;

            for(int i=0;i<36;i++){

                output += (statesSparse.col(i).conjugate().transpose() * projSparse[i] * statesSparse.col(i)).norm();

                //std::cout << (statesSparse.col(i).conjugate().transpose() * projSparse[i] * statesSparse.col(i)).norm() << std::endl;

            }

        #endif // USE_SPARSE_STATES

    #endif // USE_RK4

    return -output/36.0;

}



void MeritFunction::setIdealInverseOp(){

    Eigen::MatrixXcd SWAP(9,9);
    Eigen::MatrixXcd FULLSWAP;
    Eigen::MatrixXcd ID = Eigen::MatrixXcd::Identity(3,3);

    std::complex<double> I(0.0,1.0);

    /** === INPUT IDEAL INVERSE OPERATION ======================== */

    SWAP << 1,0,0,0,0,0,0,0,0,
            0,0,0,-I,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0,
            0,-I,0,0,0,0,0,0,0,
            0,0,0,0,1,0,0,0,0,
            0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,1,0,0,
            0,0,0,0,0,0,0,1,0,
            0,0,0,0,0,0,0,0,1;

    /** ========================================================= */

    FULLSWAP = kroneckerProduct(SWAP,ID);
    FULLSWAP = kroneckerProduct(FULLSWAP,ID);

    #ifdef USE_DENSE_STATES

        idealInverseOp = FULLSWAP;
        statesFinal = idealInverseOp.conjugate().transpose() * statesInitial;

    #endif // USE_DENSE_STATES

    #ifdef USE_SPARSE_STATES

        idealInverseOpSparse = convertToSparse(FULLSWAP);
        statesFinalSparse = idealInverseOpSparse.conjugate().transpose() * statesInitialSparse;

    #endif // USE_SPARSE_STATES

    return;

}

void MeritFunction::setProjectors(){

    Eigen::MatrixXcd basisState[6];

    Eigen::MatrixXcd outputQubitState(3,1);

    outputQubitState << 1.0,0.0,0.0;

    for(int i=0;i<6;i++) basisState[i] = Eigen::MatrixXcd::Zero(3,1);

    std::complex<double> I(0.0,1.0);

    basisState[0] << 1.0/sqrt(2.0),1.0/sqrt(2.0),0.0;
    basisState[1] << 1.0/sqrt(2.0),-1.0/sqrt(2.0),0.0;
    basisState[2] << 1.0/sqrt(2.0),I/sqrt(2.0),0.0;
    basisState[3] << 1.0/sqrt(2.0),-I/sqrt(2.0),0.0;
    basisState[4] << 1.0,0.0,0.0;
    basisState[5] << 0.0,1.0,0.0;

    proj.resize(36);

    #ifdef USE_SPARSE_STATES

        projSparse.resize(36);

    #endif // USE_SPARSE_STATES

    Eigen::MatrixXcd ID = Eigen::MatrixXcd::Identity(3,3);

    Eigen::MatrixXcd projOutputState = outputQubitState * outputQubitState.conjugate().transpose();

    int k=0;
    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){

            Eigen::MatrixXcd tempVec = kroneckerProduct(basisState[i],basisState[j]);
            Eigen::MatrixXcd tempProj = tempVec * tempVec.conjugate().transpose();

            proj[k] = kroneckerProduct(tempProj,projOutputState);
            proj[k] = kroneckerProduct(proj[k],projOutputState);


            k++;

        }
    }

    #ifdef USE_SPARSE_STATES

        for(int i=0;i<36;i++){

            projSparse[i] = convertToSparse(proj[i]);
            proj[i].resize(0,0);

        }

    #endif // USE_SPARSE_STATES

    return;

}

void MeritFunction::setInitialStates(){

    Eigen::MatrixXcd basisState[6];
    Eigen::MatrixXcd outputQubitState(3,1);

    for(int i=0;i<6;i++) basisState[i] = Eigen::MatrixXcd::Zero(3,1);

    std::complex<double> I (0.0,1.0);

    basisState[0] << 1.0/sqrt(2.0),1.0/sqrt(2.0),0.0;
    basisState[1] << 1.0/sqrt(2.0),-1.0/sqrt(2.0),0.0;
    basisState[2] << 1.0/sqrt(2.0),I/sqrt(2.0),0.0;
    basisState[3] << 1.0/sqrt(2.0),-I/sqrt(2.0),0.0;
    basisState[4] << 1.0,0.0,0.0;
    basisState[5] << 0.0,1.0,0.0;

    outputQubitState << 1.0,0.0,0.0;

    statesInitial.resize(81,36);

    #ifdef USE_DENSE_STATES

        H.initializeKMatrices(81,36);

    #endif // USE_DENSE_STATES

    #ifdef USE_SPARSE_STATES

        H.initializeKMatricesSparse(81,36);

    #endif // USE_SPARSE_STATES

    int k=0;
    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){

            Eigen::MatrixXcd tempCol = kroneckerProduct(basisState[i],basisState[j]);
            tempCol = kroneckerProduct(tempCol,outputQubitState);
            tempCol = kroneckerProduct(tempCol,outputQubitState);

            statesInitial.col(k) = tempCol;

            k++;

        }
    }

    #ifdef USE_SPARSE_STATES

        statesInitialSparse = convertToSparse(statesInitial);
        statesInitial.resize(0,0);

    #endif // USE_SPARSE_STATES

    return;

}



void MeritFunction::printReport(Eigen::VectorXd& position){

    std::cout << "OPTIMIZATION RESULT (EO =" << expansionOrder << "): "  << std::setprecision(16) << f(position) << std::endl;

    double nanCheck = -f(position);

    if(isnan(nanCheck)){

        std::cout << "THIS RESOLUTION DID NOT CONVERGE IN BFGS OPTIMIZATION" << std::endl;
        return;

    }

    std::ofstream outfile("ResolutionTest.dat",std::ofstream::app);
    outfile << expansionOrder << "\t" << std::setprecision(16) << -f(position) << std::endl;
    outfile.close();

    H.printResultHamiltonian(position,initTimeStep,finTimeStep);

    return;

}



void MeritFunction::setU(Eigen::VectorXd& position,double& eps){

    int zero = 0;

    H.evolveInfDense(zero,position,U[0]);

    UProd[0] = U[0];

    for(int i=1;i<H.timeStepsEvol;i++){

        H.evolveInfDense(i,position,U[i]);

        UProd[i] = U[i] * UProd[i-1];

    }

    for(int i=constantVariables;i<position.size();i++) position(i) += eps;

    for(int i=0;i<H.timeStepsEvol;i++){

        H.evolveInfDense(i,position,Ugrad[i]);

    }

    for(int i=constantVariables;i<position.size();i++) position(i) -= eps;

    return;

}


void MeritFunction::setMatrixExpGradient(Eigen::VectorXd& gradient, double& stepMonitor,double& eps){

    Eigen::MatrixXcd UTotal = Eigen::MatrixXcd::Identity(H.HSDimension,H.HSDimension);

    double output = 0.0;

    for(int j=0;j<=gradientGrid(0,1);j++){

        UTotal = Ugrad[j] * UTotal;

    }

    for(int j=gradientGrid(0,1)+1;j<H.timeStepsEvol;j++){

        UTotal = U[j] * UTotal;

    }

    states = UTotal * statesInitial;

    states = idealInverseOp * states;

    for(int i=0;i<36;i++){

        output += (states.col(i).conjugate().transpose() * proj[i] * states.col(i)).norm();

    }

    output /= -36.0;

    gradient(constantVariables) = -stepMonitor;
    gradient(constantVariables) += output;
    gradient(constantVariables) /= eps;


    for(int i=1;i<expansionOrder;i++){

        UTotal = UProd[gradientGrid(i-1,1)];

        output = 0.0;

        for(int j=gradientGrid(i,0);j<=gradientGrid(i,1);j++){

            UTotal = Ugrad[j] * UTotal;

        }

        for(int j=gradientGrid(i,1)+1;j<H.timeStepsEvol;j++){

            UTotal = U[j] * UTotal;

        }

        states = UTotal * statesInitial;

        states = idealInverseOp * states;

        for(int j=0;j<36;j++){

            output += (states.col(j).conjugate().transpose() * proj[j] * states.col(j)).norm();

        }

        output /= -36.0;

        gradient(constantVariables+i) = -stepMonitor;
        gradient(constantVariables+i) += output;
        gradient(constantVariables+i) /= eps;

    }

    return;

}

void MeritFunction::setGRAPEGradient(Eigen::VectorXd& position,Eigen::VectorXd& gradient){



    return;

}


MeritFunction::MeritFunction(){



}

std::complex<double> MeritFunction::sparseTrace(Eigen::SparseMatrix<std::complex<double> > M){

    std::complex<double> output(0.0,0.0);

    for(int i=0;i<M.rows();i++){

        output += M.coeff(i,i);

    }

    return output;

}

Eigen::SparseMatrix<std::complex<double> > MeritFunction::convertToSparse(Eigen::MatrixXcd& M){

    Eigen::SparseMatrix<std::complex<double> > output(M.rows(),M.cols());

    std::vector<Trip> T;

    for(int i=0;i<M.rows();i++){
        for(int j=0;j<M.cols();j++){

            if(norm(M(i,j)) > SPARSETHRESHOLD){

                T.push_back(Trip(i,j,M(i,j)));

            }

        }
    }

    output.setFromTriplets(T.begin(),T.end());

    return output;

}

Eigen::MatrixXcd MeritFunction::kroneckerProduct(Eigen::MatrixXcd& M1,Eigen::MatrixXcd& M2){

    Eigen::MatrixXcd output(M1.rows()*M2.rows(),M1.cols()*M2.cols());

    for(int i=0;i<M1.rows();i++){

        for(int j=0;j<M1.cols();j++){

            output.block(i*M2.rows(),j*M2.cols(),M2.rows(),M2.cols()) = M1(i,j) * M2;

        }

    }

    return output;

}
