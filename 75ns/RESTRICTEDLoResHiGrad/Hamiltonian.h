#ifndef HAMILTONIAN_H_INCLUDED
#define HAMILTONIAN_H_INCLUDED

//#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

typedef Eigen::Triplet<std::complex<double> > Trip;

class Hamiltonian{

    public:

        Hamiltonian();
        void setHamiltonian(int qubitDim,int NQubits,int numbConstants,int NControlSigs);
        Eigen::SparseMatrix<std::complex<double> > HBare;
        Eigen::SparseMatrix<std::complex<double> > HTotal;
        Eigen::SparseMatrix<std::complex<double> > Hk;
        void setBaseHamiltonian(Eigen::VectorXd& position);
        void setTotalHamiltonian(Eigen::VectorXd& position,int& timeStep);
        void setTimeGrid(double TINIT,double TFIN,double DTEVOL,int controlSignalResolution);
        void setInitialConstants(Eigen::VectorXd& position);
        void setInitialControlAmplitudes(Eigen::VectorXd& position);
        int timeStepsEvol;
        double timeEvol(int& timeStep);
        int HSDimension;
        void evolveStates(int& initTimeStep,int& finTimeStep,Eigen::MatrixXcd& states,Eigen::VectorXd& position);
        void evolveStatesSparse(int& initTimeStep,int finTimeStep,Eigen::SparseMatrix<std::complex<double> >& states,Eigen::VectorXd& position);
        Eigen::MatrixXcd U;
        Eigen::SparseMatrix<std::complex<double> > USparse;
        void printResultHamiltonian(Eigen::VectorXd& position,int& initTimeStep,int finTimeStep);
        void evolveInfDense(int& timeStep,Eigen::VectorXd& position,Eigen::MatrixXcd& Uinf);
        void initializeKMatrices(int rows,int cols);
        void initializeKMatricesSparse(int rows,int cols);

    private:

        int expansionOrder;
        Eigen::MatrixXcd k1,k2,k3,k4;
        Eigen::SparseMatrix<std::complex<double> > k1Sparse,k2Sparse,k3Sparse,k4Sparse;
        double doublefactorial(int x);
        Eigen::SparseMatrix<std::complex<double> > convertToSparse(Eigen::MatrixXcd& M);
        double tInit,tFin,tMid,tDisMax,dtEvol;
        int QDimension, numbQubits, constantVariables, numbControlSignals;
        std::vector<int> signalStartingPoint;
        std::complex<double> I;
        std::vector<Eigen::SparseMatrix<std::complex<double> > > m,p,n,nl;
        Eigen::SparseMatrix<std::complex<double> > HBase;
        Eigen::MatrixXcd kroneckerProduct(Eigen::MatrixXcd& M1,Eigen::MatrixXcd& M2);
        void setCreationOperators();
        void setAnnihilationOperators();
        void setNumberOperators();
        void setNonLinearTerms();
        Eigen::SparseMatrix<std::complex<double> > IDSparse,ZEROSparse;
        Eigen::MatrixXcd matrixExp(Eigen::SparseMatrix<std::complex<double> >& X);
        Eigen::SparseMatrix<std::complex<double> > matPow(Eigen::SparseMatrix<std::complex<double> >& X,int& n);
        Eigen::SparseMatrix<std::complex<double> > taylorMatrixExp(Eigen::SparseMatrix<std::complex<double> >& X);
        std::vector<Eigen::MatrixXcd> rho,lambda;
        std::vector<Eigen::SparseMatrix<std::complex<double> > > rhoSparse,lambdaSparse;
        void reverseEvolveStatesSparse(int& initTimeStep,int finTimeStep,Eigen::SparseMatrix<std::complex<double> >& states,Eigen::VectorXd& position);
        std::complex<double> sparseTrace(Eigen::SparseMatrix<std::complex<double> > M);
        void findLine(std::ifstream& infile,int resolution);

        void updateTEControlSignals(Eigen::VectorXd& position);
        void oneDimMaxSearch();
        double signalPulse(int& timeStep,int pulseNumb);
        double preSignalPulse(int& timeStep,int& pulseNumb);
        double oneDimSignalMax;
        void updateTEPreCoeffs(Eigen::VectorXd& position);
        Eigen::ArrayXXd A;
        void printSignalsAsMathematicaFunc();
        double randomDouble(double low,double high);

};


#endif // HAMILTONIAN_H_INCLUDED
