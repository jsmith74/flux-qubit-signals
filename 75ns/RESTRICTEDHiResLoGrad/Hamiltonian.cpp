#include "Hamiltonian.h"

#define SPARSETHRESHOLD 1e-30

#define PI 3.141592653589793

#define hb 0.1591549430918953

/** ========== CHOOSE INITIAL CONTROL PULSE =================== */

#define SIGNAL_MAX 3.0

#define INITAL_COEFF_SCALING 1

/** =========================================================== */


void Hamiltonian::setBaseHamiltonian(Eigen::VectorXd& position){

    HBare = position(1) * (n[3]+n[2]);
    HBare += 200.0 * (p[3] * m[2] + p[2] * m[3]);
    HBare -= 125.0 * (nl[0]+nl[1]/*+nl[2]+nl[3]*/);

    return;

}

void Hamiltonian::setInitialConstants(Eigen::VectorXd& position){

    position(0) = randomDouble(100,1000);

    position(1) = randomDouble(300.0,1200.0);

    return;

}

double Hamiltonian::randomDouble(double low,double high){

    return ((Eigen::ArrayXd::Random(1)(0)) + 1.0) * (high - low)/2.0 + low;

}

double Hamiltonian::preSignalPulse(int& timeStep,int& pulseNumb){

    double output = A(0,pulseNumb);
    double t = timeEvol(timeStep);

    for(int i=1;i<expansionOrder;i++) output += A(i,pulseNumb) * pow(t-tMid,i);

    return output;

}


void Hamiltonian::printSignalsAsMathematicaFunc(){

    std::ofstream outfile("ResolutionTest.dat",std::ofstream::app);

    for(int i=0;i<numbControlSignals;i++){

        outfile << "f[t_]:=";
        outfile << std::setprecision(16) << A(0,i);
        for(int j=1;j<expansionOrder-1;j++){
            if( A(j,i) > 0.0 ) outfile << "+" << std::setprecision(16) << A(j,i) << " * (t-" << tMid << ")^" << j;
            else outfile << std::setprecision(16) << A(j,i) << " * (t-" << tMid << ")^" << j;
        }
        if(A(expansionOrder-1,i) > 0.0) outfile << "+" << A(expansionOrder-1,i) << " * (t-" << tMid << ")^" << expansionOrder-1 << std::endl;
        else outfile << A(expansionOrder-1,i) << " * (t-" << tMid << ")^" << expansionOrder-1 << std::endl;
    }

    outfile << std::endl;

    outfile.close();

    return;

}


void Hamiltonian::oneDimMaxSearch(){



    return;

}


void Hamiltonian::setInitialControlAmplitudes(Eigen::VectorXd& position){

    int k=0;

    for(int j=0;j<numbControlSignals;j++){

        Eigen::VectorXd taylorCoeffInit = Eigen::VectorXd::Random(expansionOrder);

        for(int i=0;i<expansionOrder;i++){

            position(constantVariables+k+i) = INITAL_COEFF_SCALING * taylorCoeffInit(i);

        }

        k += expansionOrder;

    }


    return;

}


void Hamiltonian::updateTEPreCoeffs(Eigen::VectorXd& position){

    for(int j=0;j<numbControlSignals;j++){

        for(int i=2;i<expansionOrder;i++) A(i,j) = position(constantVariables+j*expansionOrder+i)/pow(tDisMax,i);

        A(1,j) = 0.0;
        for(int i=2;i<expansionOrder;i++) A(1,j) -= A(i,j) * (pow(tFin-tMid,i) - pow(tInit-tMid,i)) / (tFin - tInit);

        A(0,j) = -A(1,j) * (tInit-tMid);
        for(int i=2;i<expansionOrder;i++) A(0,j) -= A(i,j) * pow(tInit-tMid,i);

    }

    return;

}

void Hamiltonian::updateTEControlSignals(Eigen::VectorXd& position){

    updateTEPreCoeffs(position);

    oneDimMaxSearch();

    return;

}

double Hamiltonian::signalPulse(int& timeStep,int pulseNumb){

    //return preSignalPulse(timeStep,pulseNumb);
    return SIGNAL_MAX*pow(sin(preSignalPulse(timeStep,pulseNumb)),2);

}

void Hamiltonian::setTotalHamiltonian(Eigen::VectorXd& position,int& timeStep){

    HTotal = HBare + signalPulse(timeStep,0)  * (2*71.0*cos(2*PI*position(0)*timeEvol(timeStep))*(m[0] * p[3] + m[3] * p[0]));
    HTotal += signalPulse(timeStep,0) * (2*71.0*cos(2*PI*position(0)*timeEvol(timeStep))*(m[2]*p[1] + m[1]*p[2]));

    return;

}




void Hamiltonian::initializeKMatrices(int rows,int cols){

    k1.resize(rows,cols);
    k2.resize(rows,cols);
    k3.resize(rows,cols);
    k4.resize(rows,cols);

    return;

}

void Hamiltonian::initializeKMatricesSparse(int rows,int cols){

    k1Sparse.resize(rows,cols);
    k2Sparse.resize(rows,cols);
    k3Sparse.resize(rows,cols);
    k4Sparse.resize(rows,cols);

    return;

}


void Hamiltonian::evolveStatesSparse(int& initTimeStep,int finTimeStep,Eigen::SparseMatrix<std::complex<double> >& states,Eigen::VectorXd& position){

    setBaseHamiltonian(position);

    updateTEControlSignals(position);

    for(int i=initTimeStep;i<finTimeStep;i++){

        setTotalHamiltonian(position,i);

        k1Sparse = (-I/hb) * HTotal * states;
        k2Sparse = (-I/hb) * HTotal * (states + 0.5*dtEvol*k1Sparse);
        k3Sparse = (-I/hb) * HTotal * (states + 0.5*dtEvol*k2Sparse);
        k4Sparse = (-I/hb) * HTotal * (states + dtEvol*k3Sparse);

        states += (dtEvol/6.0) * (k1Sparse + 2*k2Sparse + 2*k3Sparse + k4Sparse);

        for(int j=0;j<states.cols();j++) states.col(j) = states.col(j) * (1.0/(states.col(j).norm()));

    }

    return;

}

void Hamiltonian::evolveStates(int& initTimeStep,int& finTimeStep,Eigen::MatrixXcd& states,Eigen::VectorXd& position){

    setBaseHamiltonian(position);

    for(int i=initTimeStep;i<finTimeStep;i++){

        setTotalHamiltonian(position,i);

        k1 = (-I/hb) * HTotal * states;
        k2 = (-I/hb) * HTotal * (states + dtEvol*k1/2.0);
        k3 = (-I/hb) * HTotal * (states + dtEvol*k2/2.0);
        k4 = (-I/hb) * HTotal * (states + dtEvol*k3);

        states += (dtEvol/6.0) * (k1 + 2*k2 + 2*k3 + k4);


        for(int j=0;j<states.cols();j++) states.col(j).normalize();

    }

    return;

}


void Hamiltonian::printResultHamiltonian(Eigen::VectorXd& position,int& initTimeStep,int finTimeStep){

    int zero = 0;
    std::ofstream outfile2("ResultSignal.dat");
    for(int i=initTimeStep;i<finTimeStep;i++) outfile2 << timeEvol(i) << "\t" << signalPulse(i,zero) << std::endl;
    outfile2.close();

    std::ofstream outfile("ResolutionTest.dat",std::ofstream::app);
    for(int i=0;i<constantVariables;i++) outfile << std::setprecision(16) << position(i) << "\t";
    outfile << std::endl;
    outfile << "Ending Signal:\n";
    outfile.close();
    printSignalsAsMathematicaFunc();

    return;

}


void Hamiltonian::evolveInfDense(int& timeStep,Eigen::VectorXd& position,Eigen::MatrixXcd& Uinf){

    setTotalHamiltonian(position,timeStep);

    Uinf = matrixExp(HTotal);

    return;

}



double Hamiltonian::doublefactorial(int x){

    assert(x < 171);

    double total = 1.0;

    for(int i=x;i>0;i--){
        total = i * total;
    }

    return total;

}

Eigen::MatrixXcd Hamiltonian::matrixExp(Eigen::SparseMatrix<std::complex<double> >& X){

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ces;
    ces.compute(X);

    Eigen::MatrixXcd sylvester[X.rows()];

    for(int i=0;i < X.rows();i++){
        sylvester[i] = ces.eigenvectors().col(i) * ces.eigenvectors().col(i).conjugate().transpose();
    }

    Eigen::MatrixXcd result(X.rows(),X.rows());
    result = exp(-I*dtEvol*ces.eigenvalues()(0)/hb)*sylvester[0];
    for(int j=1;j<X.rows();j++){
        result=result+exp(-I*dtEvol*ces.eigenvalues()(j)/hb)*sylvester[j];
    }

    return result;

}

void Hamiltonian::setTimeGrid(double TINIT,double TFIN,double DTEVOL,int EXPANSIONORDER){

    tInit = TINIT;
    tFin = TFIN;

    tMid = (tFin + tInit) / 2.0;

    tDisMax = (tFin - tInit) / 2.0;

    dtEvol = DTEVOL;
    timeStepsEvol = (tFin-tInit)/dtEvol;

    expansionOrder = EXPANSIONORDER;

    A.resize(expansionOrder,numbControlSignals);

    return;

}

double Hamiltonian::timeEvol(int& timeStep){

    return (timeStep + 0.5) * dtEvol + tInit;

}

void Hamiltonian::setHamiltonian(int qubitDim,int NQubits,int numbConstants,int NControlSigs){

    constantVariables = numbConstants;
    QDimension = qubitDim;
    numbQubits = NQubits;
    numbControlSignals = NControlSigs;

    HSDimension = std::pow(qubitDim,numbQubits);
    Eigen::MatrixXcd ID = Eigen::MatrixXcd::Identity(HSDimension,HSDimension);
    IDSparse = convertToSparse(ID);
    ID.resize(0,0);
    Eigen::MatrixXcd ZERO = Eigen::MatrixXcd::Zero(HSDimension,HSDimension);
    ZEROSparse = convertToSparse(ZERO);
    ZERO.resize(0,0);

    for(int i=0;i<numbQubits;i++){

        m.resize(numbQubits);
        p.resize(numbQubits);
        n.resize(numbQubits);
        nl.resize(numbQubits);

    }

    setAnnihilationOperators();

    setCreationOperators();

    setNumberOperators();

    setNonLinearTerms();

    std::complex<double> Igen(0.0,1.0);

    I = Igen;

    return;

}

Hamiltonian::Hamiltonian(){



}

void Hamiltonian::setNonLinearTerms(){

    Eigen::MatrixXcd NL = Eigen::MatrixXcd::Zero(QDimension,QDimension);
    Eigen::MatrixXcd ID = Eigen::MatrixXcd::Identity(QDimension,QDimension);

    for(int i=0;i<QDimension;i++){

        NL(i,i) = i*i - i;

    }


    for(int i=0;i<numbQubits;i++){

        Eigen::MatrixXcd NLTemp = Eigen::MatrixXcd::Identity(1,1);

        for(int j=0;j<numbQubits;j++){

            if(j==i) NLTemp = kroneckerProduct(NLTemp,NL);
            if(j!=i) NLTemp = kroneckerProduct(NLTemp,ID);

        }

        nl.at(i) = convertToSparse(NLTemp);

    }

    return;

}

void Hamiltonian::setNumberOperators(){

    Eigen::MatrixXcd N = Eigen::MatrixXcd::Zero(QDimension,QDimension);
    Eigen::MatrixXcd ID = Eigen::MatrixXcd::Identity(QDimension,QDimension);

    for(int i=0;i<QDimension-1;i++){

        N(i+1,i+1) = i+1;

    }

    for(int i=0;i<numbQubits;i++){

        Eigen::MatrixXcd NTemp = Eigen::MatrixXcd::Identity(1,1);

        for(int j=0;j<numbQubits;j++){

            if(j==i) NTemp = kroneckerProduct(NTemp,N);
            if(j!=i) NTemp = kroneckerProduct(NTemp,ID);

        }

        n.at(i) = convertToSparse(NTemp);

    }

    return;

}

void Hamiltonian::setAnnihilationOperators(){

    Eigen::MatrixXcd a = Eigen::MatrixXcd::Zero(QDimension,QDimension);
    Eigen::MatrixXcd ID = Eigen::MatrixXcd::Identity(QDimension,QDimension);

    for(int i=0;i<QDimension-1;i++){

        a(i,i+1) = sqrt(1.0*i+1.0);

    }


    for(int i=0;i<numbQubits;i++){

        Eigen::MatrixXcd mTemp = Eigen::MatrixXcd::Identity(1,1);

        for(int j=0;j<numbQubits;j++){

            if(j==i) mTemp = kroneckerProduct(mTemp,a);
            if(j!=i) mTemp = kroneckerProduct(mTemp,ID);

        }

        m.at(i) = convertToSparse(mTemp);

    }


    return;

}

void Hamiltonian::setCreationOperators(){

    for(int i=0;i<numbQubits;i++){

        p.at(i) = m.at(i).conjugate().transpose();

    }

    return;

}

Eigen::SparseMatrix<std::complex<double> > Hamiltonian::convertToSparse(Eigen::MatrixXcd& M){

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

Eigen::MatrixXcd Hamiltonian::kroneckerProduct(Eigen::MatrixXcd& M1,Eigen::MatrixXcd& M2){

    Eigen::MatrixXcd output(M1.rows()*M2.rows(),M1.cols()*M2.cols());

    for(int i=0;i<M1.rows();i++){

        for(int j=0;j<M1.cols();j++){

            output.block(i*M2.rows(),j*M2.cols(),M2.rows(),M2.cols()) = M1(i,j) * M2;

        }

    }

    return output;

}






std::complex<double> Hamiltonian::sparseTrace(Eigen::SparseMatrix<std::complex<double> > M){

    std::complex<double> output(0.0,0.0);

    for(int i=0;i<M.rows();i++){

        output += M.coeff(i,i);

    }

    return output;

}


void Hamiltonian::findLine(std::ifstream& infile,int resolution){

    std::stringstream ss;

    ss << resolution;

    std::string lineGoal;

    ss >> lineGoal;

    lineGoal = "Resolution: " + lineGoal;

    while(!infile.eof()){

        std::string line;

        std::getline(infile,line);

        if(line == lineGoal) return;

    }

    assert(1>2 && "--COULD NOT FIND RESOLUTION IN LINE SEARCH--");

    return;

}

