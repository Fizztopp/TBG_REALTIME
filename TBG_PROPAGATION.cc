/**
 *	TIGHT-BINDING MODEL FOR TWISTED BILAYER GRAPHENE (TBG)
 *  Copyright (C) 2019, Gabriel E. Topp
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 *  02111-1307, USA.
 * 	
 * 	This code uses a truncated (energy-cuttoff) Taylor-expanded Hamiltonian (A-->0) in initial band basis calculated by TBG_DOWNFOLDING.cc. 
 * 	Included are t.-d. circular gauge fields ([x,y]-plane) and a source-drain field in x-direction
 * 	The follwing objects can be calculated: 	
 *  -Time-dependent density
 *  -Observables: energy, longitudinal current, transversal current (Hall)	
 * 
 *  Necessary input:
 *  -Unit_Cell.dat: contains atomic positions, and sublattice index
 *  -BZ_FULL: List of k-points of Brilluoin zone
 */


#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <math.h>
#include <assert.h>
#include <iterator>
#include <sstream>
#include <string>
#include <algorithm>


// PARAMETERS ##########################################################

// intrinsic parameters
// electronic
#define SC        1                                                     // Defines super cell (m+1,n) and thus commensurate twist angle
#define NATOM     2                    						            // # atoms (dimension of Hamiltonian)
#define lconst    2.445                                                 // Lattice constant (Angstroem)                                        
#define	qq1       3.15													// Hopping renormalization 
#define	aa1       1.411621												// Intralayer nearest-neigbour distance	
#define	aa2       3.364                                                 // Interlayer distance (Angstroem)
#define	t1        -3.24                                                 // Hopping parameter of pz-pi (eV)
#define	t2        0.55													// Hopping parameter of pz-sigma (eV)
#define BETA      100.0                       					     	// Inverse temperature (1/eV)
// additional options
#define RG        1.0                                                   // Fermi renormalization (1. off) <-- magic angle ~1.05 <->  Natom ~13468 <-> v_fermi ~0.0
#define VV        0.0001                                                // Symmetric top-gate/back-gate potential (eV)
#define dgap      0.0001                                                // Sublattice potential a la Haldane (eV)
											 
// propagations parameters
#define starttime 0.0			                                        // Intital time											
#define endtime   1000.                                                 // End time 
#define timesteps 1e5 													// # of timesteps			
#define fac       20                                                    // Every fac-th value is stored on disc

// Peierls driving
#define w_peierls      0.2                                              // Frequency of Applied Field (in eV)
#define Ax_peierls     0.01                                             // Amplitude of Applied Field in x-direction in 1/lconst*AA
#define Ay_peierls     0.01                                             // Amplitude of Applied Field in y-direction in 1/lconst*AA
#define Az_peierls     0.0                                              // Amplitude of Applied Field in z-direction in 1/lconst*AA
#define SIGMA          100.0                                            // Sigma in 1/eV
#define DELAY          500.0                                            // Delay in 1/eV

// SOURCE DRAIN
#define ESD 0.0001                                                      // Source drain electric field strength in MV/cm (A->Int(E))
#define TSWITCH 100.                                                    // Switch on time                         

// Taylored porpagation 
#define FIRST 1.0                                                       // 1st order (1.0 on / 0.0 off)
#define SECOND 1.0                                                      // 2nd order (1.0 on / 0.0 off)
#define THIRD 1.0                                                       // 3rd order (1.0 on / 0.0 off)

// dissipation parameters
#define J0	  0.00658 											    // Coupling to occupations -> time-scale ~ 100 fs
#define J1        0.0329                                              // Coupling to coherences -> time-scale ~ 20 fs
#define BETAB     100.0                                                 // Bath temperature

#define PI 3.14159265359
#define COUNT  0


// CALCULATION OPTIONS #################################################

#ifndef NO_MPI                                                          //REMEMBER: Each Process has its own copy of all allocated memory! --> node                                             
    #include <mpi.h>
#endif

#ifndef NO_OMP                                                          // BOTTLENECK: Diagonalization -> can't be parallelized by OpenMP
    #include <omp.h>                                                    // REMEMBER: Shared memory only on same node!
#endif
                                                                                                    	
//#define NO_DISS                                                       // switch dissipation on/off             

using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name ;
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

cdouble II(0,1);

//LAPACK (Fortran 90) functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//routine to find eigensystem of Hk
extern "C" {
// computes the eigenvalues and, optionally, the left and/or right eigenvectors for HE matrices
void zheev_(char* jobz, char* uplo, int* N, cdouble* H, int* LDA, double* W, cdouble* work, int* lwork, double* rwork, int *info);
}

//INLINE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inline int fq(int i, int j, int N)
/**
 *  MAT[i,j] = Vec[fq(i,j,N)] with row index i and column index j
 */
{
    return i*N+j;
}


inline double delta(int a, int b)
/**
 *  Delta function
 */
{
	if (a==b)
		return 1.;
	else
		return 0.;
}



template <class Vec>
inline void print(Vec vec)
/**
 *	Print out vector
 */
{
	for(int i=0; i<vec.size(); i++)
		{
	    	cout << vec[i] << " ";
	    }	
	cout << endl;
}


inline double Ax_t(double &Pol, double time, dvec &ASD)
{
/**
 * Floquet Peierls field for electrons in x-direction:
 * -Pol: double to set chirality
 * -time: real time coordinate
 * -ASD: Gauge potential for constant electrical source-drain field
*/
	const double h = (endtime-starttime)/timesteps;	
	int t = int(time/h);
    return Pol*Ax_peierls*cos(w_peierls*time)*exp(-0.5*pow((time-DELAY)/SIGMA,2.))+ASD[t];
}


inline double Ay_t(double time)
{
/**
 *	Peierls field for electrons in y-direction:
 *  -time: Real time coordinate
 */
    return Ay_peierls*sin(w_peierls*time)*exp(-0.5*pow((time-DELAY)/SIGMA,2.));
}


inline double Az_t(double time)
{
/**
 *	Peierls field for electrons in z-direction:
 *  -time: real time coordinate
 */
    return Az_peierls*sin(w_peierls*time)*exp(-0.5*pow((time-DELAY)/SIGMA,2.));
}


inline double fermi(double energy, double mu)
{
/**
 *	Fermi distribution:
 *	-energy: Energy eigenvalue
 *	-mu: Chemical potential
 */
    return 1./(exp((energy-mu)*BETA) + 1.);
}


inline double gauss(double time, double delay, double sigma)
/**
 *	Gauss distribution function
 *	-time: real time coordinate
 * 	-delay: shift of mean value
 * 	-sigma: sigma value
 */
{
	return 1./(sigma*sqrt(2.*PI))*exp(-0.5*pow((time-delay)/sigma,2.));
}


// VOID FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void ReadIn(vector<dvec> &MAT, const string& filename)
{
/**
 *	Read in real valued matrix
 */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		dvec row((istream_iterator<double>(is)),
		istream_iterator<double>());
		MAT.push_back(row);
	}
	in.close();
}


template <class Vec>
void times(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product of quadratic matrices: $C = A \cdot B$
 */
{
    int dim = sqrt(A.size());
	Vec TEMP(dim*dim);
    // Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
	    for(int j=0; j<dim; j++) {
		    TEMP[fq(j,i,dim)] = B[fq(i,j,dim)];
		   }
    }
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += A[fq(i,k,dim)]*TEMP[fq(j,k,dim)]; 
			}
		}
	}	
}


template <class Vec>
void times_dn(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of first factor: $C = A^\dagger \cdot B$
 */
{
	int dim = sqrt(A.size());
	Vec TEMP1(dim*dim);
	Vec TEMP2(dim*dim);
	// Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
		for(int j=0; j<dim; j++) {
			TEMP1[fq(j,i,dim)] = A[fq(i,j,dim)];
			TEMP2[fq(j,i,dim)] = B[fq(i,j,dim)];
		}
	}		
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += conj(TEMP1[fq(i,k,dim)])*TEMP2[fq(j,k,dim)];
			}
		}
	}		
}


template <class Vec>
void times_nd(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of second factor: $C = A \cdot B^\dagger$
 */
{
	int dim = sqrt(A.size());	
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
					C[fq(i,j,dim)] += A[fq(i,k,dim)]*conj(B[fq(j,k,dim)]);
			}
		}
	}	
}


void set_Hk_DOWN(int &dim_new, cvec &Hk_DOWN, vector<cvec*> Hk_DOWN_LIST, dvec &ASD, double &Pol, double time)
/**
  *	Set downfolded td Hamiltonian 
  * -dim_new: integer value of reduced leading order of Hamiltonian
 *  -Hk_DOWN: Complex vector[dim_new x dim_new] to store Hamiltonian matrix
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  * -ASD: Gauge field of source-drain field
  * -Pol: double to set chirality
  * -time: tiome variable
  */
{
	double AX = Ax_t(Pol,time,ASD)/lconst;
	double AY = Ay_t(time)/lconst;
	
	for(int i=0; i<dim_new*dim_new; ++i){
		Hk_DOWN[i] = (*Hk_DOWN_LIST[0])[i] + FIRST*((*Hk_DOWN_LIST[1])[i]*AX + (*Hk_DOWN_LIST[2])[i]*AY) + SECOND*1./2.*((*Hk_DOWN_LIST[3])[i]*AX*AX + 2.*(*Hk_DOWN_LIST[4])[i]*AX*AY + (*Hk_DOWN_LIST[5])[i]*AY*AY) + THIRD*1./6.*((*Hk_DOWN_LIST[6])[i]*AX*AX*AX + 3.*(*Hk_DOWN_LIST[7])[i]*AX*AX*AY + 3.*(*Hk_DOWN_LIST[8])[i]*AX*AY*AY + (*Hk_DOWN_LIST[9])[i]*AY*AY*AY); 		
	}	
}	


void set_dHkdAx_DOWN(int &dim_new, cvec &Hk_DOWN, vector<cvec*> Hk_DOWN_LIST, dvec &ASD, double &Pol, double time)
/**
  * Set downfolded t.-d. derivative by Ax of Hamiltonian 
  * -dim_new: integer value of reduced leading order of Hamiltonian
 *  -Hk_DOWN: Complex vector[dim_new x dim_new] to store Hamiltonian matrix
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  * -ASD: Gauge field of source-drain field
  * -Pol: double to set chirality
  * -time: tiome variable
  */
{
	double AX = Ax_t(Pol,time,ASD)/lconst;
	double AY = Ay_t(time)/lconst;
	
	for(int i=0; i<dim_new*dim_new; ++i){
		Hk_DOWN[i] =  FIRST*(*Hk_DOWN_LIST[1])[i] + SECOND*1./2.*(2.*(*Hk_DOWN_LIST[3])[i]*AX + 2.*(*Hk_DOWN_LIST[4])[i]*AY) + THIRD*1./6.*(3.*(*Hk_DOWN_LIST[6])[i]*AX*AX + 2.*3.*(*Hk_DOWN_LIST[7])[i]*AX*AY + 3.*(*Hk_DOWN_LIST[8])[i]*AY*AY); 	
	}		
}	


void set_dHkdAy_DOWN(int &dim_new, cvec &Hk_DOWN, vector<cvec*> Hk_DOWN_LIST, dvec &ASD, double &Pol, double time)
/**
  * Set downfolded t.-d. derivative by Ay of Hamiltonian 
  * -dim_new: integer value of reduced leading order of Hamiltonian
  * -Hk_DOWN: Complex vector[dim_new x dim_new] to store Hamiltonian matrix
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  * -ASD: Gauge field of source-drain field
  * -Pol: double to set chirality
  * -time: tiome variable
  */
{
	double AX = Ax_t(Pol,time,ASD)/lconst;
	double AY = Ay_t(time)/lconst;

	for(int i=0; i<dim_new*dim_new; ++i){
		Hk_DOWN[i] = FIRST*(*Hk_DOWN_LIST[2])[i] + SECOND*1./2.*(2.*(*Hk_DOWN_LIST[4])[i]*AX + 2.*(*Hk_DOWN_LIST[5])[i]*AY) + THIRD*1./6.*(3.*(*Hk_DOWN_LIST[7])[i]*AX*AX + 2.*3.*(*Hk_DOWN_LIST[8])[i]*AX*AY + 3.*(*Hk_DOWN_LIST[9])[i]*AY*AY);  		
	}		
}	


void diagonalize_DOWN(cvec &Hk, dvec &evals_DOWN, int &dim_new)
{
/**
 *  Diagonalization of matrix Hk. Writes eiegenvalues to vector evals and eigenvectors (normalized!) to matrix Hk
 *  -Hk: Complex vector[dim_new x dim_new] to store Hamiltonian --> transformation matrices
 * 	-evals_DOWN: Real vector[dim_new] to store eigenvalues
 *  -dim_new: integer value of reduced leading order of Hamiltonian
 */
	char    jobz = 'V';             									//'N','V':  Compute eigenvalues only/+eigenvectors
	char    uplo = 'U';              									//'U','L':  Upper/Lower triangle of H is stored
	int     matsize = dim_new;      									// The order of the matrix A.  N >= 0
	int     lda = dim_new;            									// The leading dimension of the array A.  LDA >= max(1,N)
	int     lwork = 2*dim_new-1;      									// The length of the array WORK.  LWORK >= max(1,2*N-1)
	double  rwork[3*dim_new-2];       									// dimension (max(1, 3*N-2))
	cdouble work[2*dim_new-1];        									// dimension (MAX(1,LWORK)) On exit, if INFO = 0, WORK(1) returns the optimal LWORK
	int	    info;
	zheev_(&jobz, &uplo, &matsize, &Hk[0], &lda, &evals_DOWN[0], &work[0], &lwork, &rwork[0], &info);
	assert(!info);
}	


void set_dRHOdt_DOWN(int &dim_new, cvec &TEMP1, cvec &TEMP2, cvec &RHO_t_tk, cvec &dRHO_dt, cvec &Hk_DOWN, dvec &evals_DOWN, double mu)
/**
 *  Calculation of the time-derivative of the density matrix
 *  -dim_new: integer value of reduced leading order of Hamiltonian
 *  -TEMP1, TEMP2: Complex helper matrix 
 *  -RHO_t_tk: Complex vector[dim_new x dim_new] of k- and time-dependent density matrix
 *  -dRHO_dt: Complex vector[dim_new x dim_new] of temporal change of density matrix
 *  -Hk_DOWN: Complex vector[dim_new x dim_new] to store Hamiltonian matrix
 * 	-evals_DOWN: Real vector[dim_new] to store eigenvalues
 *  -mu: chemical potential
 */
{	
	// COHERENT PART
	times(Hk_DOWN, RHO_t_tk, TEMP1);										
	times(RHO_t_tk, Hk_DOWN, TEMP2);

#ifndef NO_OMP    	
	#pragma omp parallel for                                 
#endif		 			
	for(int i=0; i<dim_new*dim_new; i++)
	{
		dRHO_dt[i] = -II*(TEMP1[i]-TEMP2[i]);	
	}
	
	// DISSIPATION PART	
#ifndef NO_DISS
    diagonalize_DOWN(Hk_DOWN, evals_DOWN, dim_new);

	for(int i=0; i<dim_new*dim_new; i++)
		TEMP1[i] = RHO_t_tk[i];											
	
    // transform Rho: -> t.d. band basis
	times_nd(TEMP1, Hk_DOWN, TEMP2);
	times(Hk_DOWN, TEMP2, TEMP1);										
	
	// calculate dRho
	for(int a=0; a<dim_new; a++)
	{
		for(int b=0; b<dim_new; b++)
		{
			TEMP2[fq(a,b,dim_new)] = -J0*(TEMP1[fq(a,b,dim_new)]-fermi(evals_DOWN[a], mu))*delta(a,b) - J1*TEMP1[fq(a,b,dim_new)]*(1.-delta(a,b));
		}
	}		
	
	// transform dRho: -> intitial band basis
	times(TEMP2, Hk_DOWN, TEMP1);                                        
	times_dn(Hk_DOWN, TEMP1, TEMP2);			
	
	for(int i=0; i<dim_new*dim_new; i++)
	{
		dRHO_dt[i] += TEMP2[i];
	}	
#endif 		
}


void ReadInMAT(int &dim_new, vector<cvec*> Hk_DOWN_LIST, const string& filename)
{
/**
  * Read in taylore matrices from disc
  * -dim_new: integer value of reduced leading order of Hamiltonian
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  *	-filename: String to define file
  */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		cvec row((istream_iterator<cdouble>(is)),	
		istream_iterator<cdouble>());
		//cout << row.size() << " " << dim_new << "---------------------------------------------------------------------" << endl;
		for(int m=0; m<10; ++m)	
		{	
			for(int i=0; i<dim_new*dim_new; ++i)
			{
				(*Hk_DOWN_LIST[m])[i] = row[fq(m,i,dim_new*dim_new)];
				//cout << row[fq(m,i,dim_new*dim_new)];
			}
		}	

	}
	in.close();
}


void AB2_propatation_DOWN(double &mu, double &delta_mu, vector<dvec> &BZ_FULL, dvec &ASD, double &Pol, int num, int &numprocs, int &myrank)
/**
 *	Two-step Adams-Bashforth linear multistep propagator:
 *  -mu: chemical potential
 *  -delta_mu: change in chemical potential to adjust filling
 *	-BZ_FULL: k-points of reciprocal cell
 *  -ASD: Gauge field of source-drain field
 *  -Pol: double to set chirality
 *  -nu: int number to tag file for storage
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI) 
 */
{
	const int num_kpoints_BZ_full = BZ_FULL.size();                     // # of k-vectors from sampling of irreducible BZ
	const double h = (endtime-starttime)/timesteps;	
	
	int dim_new, count;
	vector<int> limits(2);
	vector<int> dimensions(num_kpoints_BZ_full);
	dvec kvec;
	dvec E_t(timesteps, 0.);
	dvec E_0(timesteps, 0.);
	dvec N_t(timesteps, 0.);
	dvec C_t(2*timesteps, 0.);
	dvec C_0(2*timesteps, 0.);
	cvec *temp0, *temp1, *temp2;
	ifstream input;

	// Read in dimension of matrices
	input.open("../DIMENSIONS.dat");
	
	for(int k=0; k<num_kpoints_BZ_full; ++k)
	{
		 input >> dimensions[k];
		 //cout << dimensions[k] << " " << endl;
	}
	input.close();
	
	count = 0;
	// Propagation
	for(int k=myrank; k<num_kpoints_BZ_full; k+=numprocs)	
	{   
		// Set k-vector
		
		kvec = BZ_FULL[k];	
		dim_new = dimensions[k];
		if(dim_new==0){
			continue;
		}
		//if(myrank==0) cout <<  dim_new << "------------------------------------------------------------------------------------------------------" << endl;
		
		// Dynamical allocation of memory for tuncated taylor matrices
		vector<cvec*> Hk_DOWN_LIST(10);
		for(int m=0; m<10; m++)
			Hk_DOWN_LIST[m] = new cvec(dim_new*dim_new);	

		// Read in tuncated taylor matrices from file
		ReadInMAT(dim_new, Hk_DOWN_LIST, "../HK_DOWN_LIST/HK_DOWN_LIST_"+to_string(k)+".dat");

		// Dynamical allocation of memory for tuncated Hamiltonian    
		cvec *Hk_DOWN = new cvec(dim_new*dim_new);
		
		// Dynamical allocation of memory for RHO[k,t]
		vector<cvec*> RHO_t(3);                                           
		for(int t=0; t<3; t++)
			RHO_t[t] = new cvec(dim_new*dim_new);	
			
		// Dynamical allocation of memory for dRHO_dt[k,t]
		vector<cvec*> dRHO_dt(3);                                           
		for(int t=0; t<3; t++)
			dRHO_dt[t] = new cvec(dim_new*dim_new);	                                		

		// Dynamical allocation of memory for helper matrices
		cvec *RHO_0 = new cvec(dim_new*dim_new);
		cvec *SMAT_0 = new cvec(dim_new*dim_new);
		cvec *TEMP1 = new cvec(dim_new*dim_new);									// Helper arrays for set_dRhodt()
		cvec *TEMP2 = new cvec(dim_new*dim_new);
		cvec *TEMP3 = new cvec(dim_new*dim_new);
		cvec *TEMP4 = new cvec(dim_new*dim_new);
		cvec *TEMP5 = new cvec(dim_new*dim_new);
		cvec *TEMP6 = new cvec(dim_new*dim_new);
		dvec *evals_DOWN = new dvec(dim_new);

		//if(myrank==0)cout << "HALLO_05" << " --------------------------------------------------------------------------------------------------------" << endl;
		for(int i=0; i<dim_new; i++)			
		{
			for(int j=0; j<dim_new; j++){
				(*RHO_t[0])[fq(i, j, dim_new)] = delta(i,j)*fermi(real((*Hk_DOWN_LIST[0])[fq(i, j, dim_new)]), mu+delta_mu); 
				(*RHO_0)[fq(i, j, dim_new)] = (*RHO_t[0])[fq(i, j, dim_new)];
			}	
		}
		// Calculation of inital energy	
		set_Hk_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, 0.0);
		times(RHO_t[0][0], Hk_DOWN[0], TEMP1[0]);
		// Calculation of inital c
		set_dHkdAx_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, 0.0);
		times(RHO_t[0][0], Hk_DOWN[0], TEMP2[0]);
		set_dHkdAx_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, 0.0);
		times(RHO_t[0][0], Hk_DOWN[0], TEMP3[0]);
		// Store information
		for(int i=0; i<dim_new; i++)
		{
			E_t[0] += real((*TEMP1)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
			E_0[0] += real((*TEMP1)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
			C_t[fq(0,0,timesteps)] += real((*TEMP2)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);	
			C_t[fq(1,0,timesteps)] += real((*TEMP3)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
			C_0[fq(0,0,timesteps)] += real((*TEMP2)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);	
			C_0[fq(1,0,timesteps)] += real((*TEMP3)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
			N_t[0] += real((*RHO_t[0])[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
		}
		for(int t=0; t<timesteps-1; t++)
		{
			// 1st Euler step	
			if(t==0)
			{
				set_Hk_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t));
				set_dRHOdt_DOWN(dim_new, TEMP1[0], TEMP2[0], RHO_t[0][0], dRHO_dt[0][0], Hk_DOWN[0], evals_DOWN[0], mu+delta_mu);
#ifndef NO_OMP    	
	#pragma omp parallel for                                 
#endif						
				for(int i=0; i<dim_new*dim_new; i++)
				{
					(*RHO_t[1])[i] = (*RHO_t[0])[i] + h*(*dRHO_dt[0])[i]; 		
				}	
				// Calculation of total energy	
				set_Hk_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				times(RHO_t[1][0], Hk_DOWN[0], TEMP1[0]);
				times(RHO_0[0], Hk_DOWN[0], TEMP6[0]);
				// Calculation of inital c
				set_dHkdAx_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				times(RHO_t[1][0], Hk_DOWN[0], TEMP2[0]);
				times(RHO_0[0], Hk_DOWN[0], TEMP4[0]);
				set_dHkdAy_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				times(RHO_t[1][0], Hk_DOWN[0], TEMP3[0]);
				times(RHO_0[0], Hk_DOWN[0], TEMP5[0]);
				for(int i=0; i<dim_new; i++)
				{
					E_t[t+1] += real((*TEMP1)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					E_0[t+1] += real((*TEMP6)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					C_t[fq(0,t+1,timesteps)] += real((*TEMP2)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);	
					C_t[fq(1,t+1,timesteps)] += real((*TEMP3)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					C_0[fq(0,t+1,timesteps)] += real((*TEMP4)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);	
					C_0[fq(1,t+1,timesteps)] += real((*TEMP5)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);					
					N_t[t+1] += real((*RHO_t[1])[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
				}			
			}
			// Two step Adams–Bashforth method
			else
			{	// 2-step Adams predictor	
				set_Hk_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t));
				set_dRHOdt_DOWN(dim_new, TEMP1[0], TEMP2[0], RHO_t[1][0], dRHO_dt[1][0], Hk_DOWN[0], evals_DOWN[0], mu+delta_mu);
#ifndef NO_OMP    	
	#pragma omp parallel for                                 
#endif								
				for(int i=0; i<dim_new*dim_new; i++)
				{
					(*RHO_t[2])[i] = (*RHO_t[1])[i] + h*(3./2.*(*dRHO_dt[1])[i] - 0.5*(*dRHO_dt[0])[i]); 		// P_{n+1} = y_{n} + 3/2*h*f(t_{n},y_{n}) - 0.5*h*f(t_{n-1},y_{n-1})
				}	
				// 2-step Moulton corrector
				set_Hk_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				set_dRHOdt_DOWN(dim_new, TEMP1[0], TEMP2[0], RHO_t[2][0], dRHO_dt[2][0], Hk_DOWN[0], evals_DOWN[0], mu+delta_mu);
#ifndef NO_OMP    	
	#pragma omp parallel for                                 
#endif							
				for(int i=0; i<dim_new*dim_new; i++)
				{
					(*RHO_t[2])[i] = (*RHO_t[1])[i] + 0.5*h*((*dRHO_dt[2])[i] + (*dRHO_dt[1])[i]); 		        // y_{n+1} = y_{n} + 1/2*h*(f(t_{n+1},P_{n+1}) + f(t_{n},y_{n}))
				}
				// calculation of total energy	
				set_Hk_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				times(RHO_t[2][0], Hk_DOWN[0], TEMP1[0]);
				times(RHO_0[0], Hk_DOWN[0], TEMP6[0]);
				// calculation of inital c
				set_dHkdAx_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				times(RHO_t[2][0], Hk_DOWN[0], TEMP2[0]);				
				times(RHO_0[0], Hk_DOWN[0], TEMP4[0]);
				set_dHkdAy_DOWN(dim_new, Hk_DOWN[0], Hk_DOWN_LIST, ASD, Pol, h*double(t+1));
				times(RHO_t[2][0], Hk_DOWN[0], TEMP3[0]);
				times(RHO_0[0], Hk_DOWN[0], TEMP5[0]);
				for(int i=0; i<dim_new; i++)
				{
					E_t[t+1] += real((*TEMP1)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					E_0[t+1] += real((*TEMP6)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					C_t[fq(0,t+1,timesteps)] += real((*TEMP2)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);	
					C_t[fq(1,t+1,timesteps)] += real((*TEMP3)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					C_0[fq(0,t+1,timesteps)] += real((*TEMP4)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);	
					C_0[fq(1,t+1,timesteps)] += real((*TEMP5)[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
					N_t[t+1] += real((*RHO_t[2])[fq(i,i,dim_new)])/double(num_kpoints_BZ_full);
				}
				temp0 = RHO_t[0];
				temp1 = RHO_t[1];
				temp2 = RHO_t[2]; 
				RHO_t[0] = temp1;
				RHO_t[1] = temp2;
				RHO_t[2] = temp0;
				
				temp0 = dRHO_dt[0];
				temp1 = dRHO_dt[1];
				temp2 = dRHO_dt[2]; 
				dRHO_dt[0] = temp1;
				dRHO_dt[1] = temp2;
				dRHO_dt[2] = temp0;
			}				
			//if(myrank==0) 
			//{	
			//	cout << "loop #" << count*(timesteps-2)+t << " of " << double(timesteps-2)*double(num_kpoints_BZ_full)/double(numprocs) << " --> " << 100.*(count*(timesteps-2)+t)/(double(timesteps-2)*double(num_kpoints_BZ_full)/double(numprocs)) << " %" << endl;
			//	cout << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
			//}					
		}
		count++;
		delete TEMP1, TEMP2, TEMP3, TEMP4, TEMP5, TEMP6, RHO_0, SMAT_0, evals_DOWN, Hk_DOWN; 
		for(int m=0; m<3; m++)
		{                            
			delete RHO_t[m];
			delete dRHO_dt[m];
		}	
		for(int m=0; m<10; m++)
			delete Hk_DOWN_LIST[m];
	}
#ifndef NO_MPI		
	MPI_Allreduce(MPI_IN_PLACE, &E_t[0], timesteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &E_0[0], timesteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &N_t[0], timesteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
	MPI_Allreduce(MPI_IN_PLACE, &C_t[0], 2*timesteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &C_0[0], 2*timesteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);		
#endif	
	// Write td mean field parameters to file
	string POL; 
	if(Pol>0.0)
		POL = 'L';
	else
		POL = 'R';
	if(myrank==0)
	{
		ofstream myfile ("DATA/200/E_t"+POL+to_string(num)+".dat");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps; t+=fac)
			{
				//myfile << E_t[t]-E_0[t] << endl;
				myfile << E_t[t] << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
	}
	if(myrank==0)
	{
		ofstream myfile ("DATA/200/n_t"+POL+to_string(num)+".dat");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps; t+=fac)
			{   
				myfile  << N_t[t] << endl;
			}	
			myfile.close();
		}
    else cout << "Unable to open file" << endl;
	}
	if(myrank==0)
	{
		ofstream myfile ("DATA/200/C_t"+POL+to_string(num)+".dat");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps; t+=fac)
			{   
				myfile << C_t[fq(0,t,timesteps)]-C_0[fq(0,t,timesteps)] << " " << C_t[fq(1,t,timesteps)]-C_0[fq(1,t,timesteps)] << endl;
				//myfile << C_t[fq(0,t,timesteps)] << " " << C_t[fq(1,t,timesteps)] << endl;
			}	
			myfile.close();
		}
    else cout << "Unable to open file" << endl;
	}
}


// main() function #####################################################

int main(int argc, char * argv[])
{
    //************** MPI INIT ***************************
  	int numprocs=1, myrank=0, namelen;
    
#ifndef NO_MPI
  	char processor_name[MPI_MAX_PROCESSOR_NAME];
  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	MPI_Get_processor_name(processor_name, &namelen);
    
	cout << "Process " << myrank << " on " << processor_name << " out of " << numprocs << " says hello." << endl;
	MPI_Barrier(MPI_COMM_WORLD);
    
#endif
	if(myrank==0) cout << "\n\tProgram running on " << numprocs << " processors." << endl;

	//************** OPEN_MP INIT **************************************
#ifndef NO_OMP 	  
	cout << "# of processes " << omp_get_num_procs() << endl;
#pragma omp parallel 
	cout << "Thread " << omp_get_thread_num() << " out of " << omp_get_num_threads() << " says hello!" << endl;     
#endif
	//******************************************************************
   
	// DECLARATION AND INTITALIZATIO
	const int a = SC+1;
	const int b = SC;

	if(NATOM != 4*(SC*SC+(SC+1)*SC+(SC+1)*(SC+1)))
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}
	
	// 1st angle   
	const double angle1 = atan2(double(b)*sqrt(3.)/2.,double(a)+double(b)/2.) ;
	if(myrank==0) cout << "agle1: " << angle1 << endl;
	// 2nd angle       
	const double angle2 = angle1 + PI/3. ;                                          
	if(myrank==0) cout << "agle2: " << angle2 << endl;
	
	// side length of super cell
	const double d = sqrt(double(b*b)*3./4.+pow(double(a)+double(b)/2.,2.));
	if(myrank==0) cout << "d: " << d << endl;
	
	// superlattice bravis translational vectors
	const dvec lvec = {d*cos(angle1),  d*sin(angle1), d*sin(PI/6.-angle1), d*cos(PI/6.-angle1)};
	
	// chemical potential
	double mu;
	
	//Read in vector of k-points
	vector<dvec> K_PATH;
	ReadIn(K_PATH, "../k_path.dat");
	if(myrank==0) cout << "high-symmetry path --> " << K_PATH.size() << " points" << endl;
	int num_kpoints_PATH = K_PATH.size();
	
	// irr. BZ
	//vector of weights
	vector<dvec> kweights_irr;
	ReadIn(kweights_irr, "../k_weights_irr.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_IRR;
	ReadIn(BZ_IRR, "../k_BZ_irr.dat");
	if(myrank==0) cout << "irreducible BZ --> " << BZ_IRR.size() << " points" << endl;
	int num_kpoints_BZ = BZ_IRR.size();
	
    // full BZ
	//vector of weights
	vector<dvec> kweights_full;
	ReadIn(kweights_full, "../k_weights_full.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_FULL;
	ReadIn(BZ_FULL, "../k_BZ_full.dat");
	if(myrank==0) cout << "full BZ --> " << BZ_FULL.size() << " points" << endl;
	int num_kpoints_BZ_full = BZ_FULL.size();
	
	// PATCH
	//vector of weights
	vector<dvec> kweights_PATCH;
	ReadIn(kweights_PATCH, "../k_weights_PATCH.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_PATCH;
	ReadIn(BZ_PATCH, "../k_BZ_PATCH.dat");
	if(myrank==0) cout << "PATCH --> " << BZ_PATCH.size() << " points" << endl;
	int num_kpoints_BZ_PATCH = BZ_PATCH.size();
	
	// Source-Drain field
	dvec EE(timesteps);
	dvec ASD(timesteps);
	dvec TIME(timesteps);
	
	const double h = (endtime-starttime)/timesteps;	
	
	// Set E-field
	for(int tt=0; tt<timesteps; tt++)
	{
		TIME[tt] = double(tt)*h;
		if(TSWITCH < TIME[tt]){
			EE[tt] = ESD;
		} 
		else if(starttime < TIME[tt]<= TSWITCH){
			EE[tt] = ESD*(3.*pow(TIME[tt]/TSWITCH,2.)-2.*pow(TIME[tt]/TSWITCH,3.));
		}
		else{
			EE[tt] = 0.;
		}	
		EE[tt] = EE[tt]*lconst*0.01;                                    // [A]=lconst^(-1): Int[EE*MV/cm*dt*1/eV] = Int[EE*dt]*MV/cm*1/eV = 1e6/1e8*1/AA = 1e-2*lconst/a0
	}	
	// Set A-field
	ASD[0] = 0.0;
	for(int tt=0; tt<timesteps-1; tt++)
	{	
		ASD[tt+1] = ASD[tt]+0.5*(EE[tt+1]+EE[tt])*h;  
	}	
	if(myrank==0)
	{
		ofstream myfile ("DATA/200/ASD_t.dat");
		if (myfile.is_open())
		{
			for(int tt=0; tt<timesteps; tt++)
			{
				myfile << EE[tt] << " " << ASD[tt] << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
	}	
		
	int dim_new;

	// CALCULATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	const clock_t begin_time = clock();                                 // time summed over all threads
#ifndef NO_OMP	 
	double dtime = omp_get_wtime();	                                    // time per core
#endif
	
	// load chemical potential from file
	ifstream fin("../mu.dat");
    fin >> mu;
    if(myrank==0) cout << "Initial chemical potential: " << mu << endl;
	
	// Run propagation
	double LEFT = 1.0;
	double RIGHT = -1.0;
	double delta_mu = -0.2;
	
	for(int i=0; i<40; i++)
	{
		AB2_propatation_DOWN(mu, delta_mu, BZ_FULL, ASD, LEFT, i, numprocs, myrank);
		AB2_propatation_DOWN(mu, delta_mu, BZ_FULL, ASD, RIGHT, i, numprocs, myrank);
		delta_mu = delta_mu+0.01;
		if(myrank==0) cout << "Detla mu: " << delta_mu << " -------------------------------------------------------------------------" << endl;
	}
	
	if(myrank==0)
	{ 
		cout << "Calculations time (MPI): " << float(clock() - begin_time)/CLOCKS_PER_SEC << " seconds" << endl;
#ifndef NO_OMP	
	dtime = omp_get_wtime() - dtime;
	cout << "Calculations time (OMP): " << dtime << " seconds" << endl; 
#endif	
	}
	
#ifndef NO_MPI
	MPI_Finalize();
#endif	

}


