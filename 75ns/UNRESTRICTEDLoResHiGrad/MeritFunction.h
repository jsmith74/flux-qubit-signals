#ifndef MERITFUNCTION_H_INCLUDED
#define MERITFUNCTION_H_INCLUDED


#include "Hamiltonian.h"


class MeritFunction{

    public:

        MeritFunction();
        void setMeritFunction(int intParam);
        double f(Eigen::VectorXd& position);
        int funcDimension,constantVariables;
        void printReport(Eigen::VectorXd& position);
        Eigen::VectorXd setInitialPosition();
        void setGRAPEGradient(Eigen::VectorXd& position,Eigen::VectorXd& gradient);
        void setU(Eigen::VectorXd& position,double& eps);
        void setMatrixExpGradient(Eigen::VectorXd& gradient, double& stepMonitor,double& eps);

    private:

        int expansionOrder,numbControlSignals;
        int printCounter;
        Eigen::SparseMatrix<std::complex<double> > convertToSparse(Eigen::MatrixXcd& M);
        Eigen::MatrixXcd kroneckerProduct(Eigen::MatrixXcd& M1,Eigen::MatrixXcd& M2);
        void setInitialStates();
        void setIdealInverseOp();
        void setProjectors();
        std::vector<Eigen::MatrixXcd> proj;
        std::vector<Eigen::SparseMatrix<std::complex<double> > > projSparse;
        Eigen::MatrixXcd statesInitial,statesFinal,states,idealInverseOp;
        Eigen::SparseMatrix<std::complex<double> > statesInitialSparse,statesFinalSparse,statesSparse,idealInverseOpSparse;
        int initTimeStep,finTimeStep;
        double tInit,tFin,dt;
        Hamiltonian H;
        std::complex<double> sparseTrace(Eigen::SparseMatrix<std::complex<double> > M);
        std::vector<Eigen::MatrixXcd> U,Ugrad,UProd;
        Eigen::MatrixXi gradientGrid;

};

#endif // MERITFUNCTION_H_INCLUDED
