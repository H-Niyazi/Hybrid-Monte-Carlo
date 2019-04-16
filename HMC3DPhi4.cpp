// Hossein Niyazi, April 2019
// HMC for a (2+1)D phi^4 QFT

#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
using namespace std;

double BoxMuller()
{
  double x1 = drand48();
  double x2 = drand48();
  return sqrt(-2*log(1-x1))*cos(2*M_PI*(1-x2));
}

struct site { int x, y, t, index; };

struct params
{
  int Nx, Ny, Nt, volume;
  double kappa, lambda;
};

int idx ( int x, int y, int t, int nx, int ny, int nt )
{ return x + y*nx + t*nx*ny; }

int posx ( int index, int nx )
{ return index % nx; }

int posy ( int index, int nx, int ny )
{ return (index % (nx*ny)) / nx; }

int post ( int index, int nx, int ny )
{ return index / (nx*ny); }

double action(double* phi, site* lattice, params param)
{
  double kappa  = param.kappa;
  double lambda = param.lambda;
  int volume    = param.volume;
  int nx        = param.Nx;
  int ny        = param.Ny;
  int nt        = param.Nt;

  double S = 0.;
  int next_neighbor_index;
  int x, y, t;

  for (int i = 0; i < volume; i++)
  {
    S += phi[i]*phi[i] + lambda*(phi[i]*phi[i]-1.0)*(phi[i]*phi[i]-1.0);

    x = lattice[i].x;
    y = lattice[i].y;
    t = lattice[i].t;

    next_neighbor_index = idx((x+1)%nx,y,t,nx,ny,nt);
    S += -2.*kappa*phi[i]*phi[next_neighbor_index];

    next_neighbor_index = idx(x,(y+1)%ny,t,nx,ny,nt);
    S += -2.*kappa*phi[i]*phi[next_neighbor_index];

    next_neighbor_index = idx(x,y,(t+1)%nt,nx,ny,nt);
    S += -2.*kappa*phi[i]*phi[next_neighbor_index];
  }

  return S;
}

double Hamiltonian(double* phi, double* p, site* lattice, params param)
{
  int volume = param.volume;

  double H = action(phi, lattice, param);
  for (int i = 0; i < volume; i++)
    H += 0.5 * p[i] * p[i];

  return H;
}

double force(int i, double* phi, site* lattice, params param)
{
  double kappa  = param.kappa;
  double lambda = param.lambda;
  int volume    = param.volume;
  int nx        = param.Nx;
  int ny        = param.Ny;
  int nt        = param.Nt;

  double f = 0.;
  int next_neighbor_index;
  int prev_neighbor_index;
  int x, y, t;

  f += 2. * phi[i] + 4. * lambda * ( phi[i] * phi[i] - 1.0 ) * phi[i];

  x = lattice[i].x;
  y = lattice[i].y;
  t = lattice[i].t;

  next_neighbor_index = idx((x+1)%nx,y,t,nx,ny,nt);
  prev_neighbor_index = idx((x-1+nx)%nx,y,t,nx,ny,nt);
  f += -2. * kappa * phi[next_neighbor_index];
  f += -2. * kappa * phi[prev_neighbor_index];

  next_neighbor_index = idx(x,(y+1)%ny,t,nx,ny,nt);
  prev_neighbor_index = idx(x,(y-1+ny)%ny,t,nx,ny,nt);
  f += -2. * kappa * phi[next_neighbor_index];
  f += -2. * kappa * phi[prev_neighbor_index];

  next_neighbor_index = idx(x,y,(t+1)%nt,nx,ny,nt);
  prev_neighbor_index = idx(x,y,(t-1+nt)%nt,nx,ny,nt);
  f += -2. * kappa * phi[next_neighbor_index];
  f += -2. * kappa * phi[prev_neighbor_index];

  return -f;
}

int leapfrog_update(double* phi, site* lattice, params param,
                    double traj, int Ns)
{
  int volume = param.volume;
  int nx     = param.Nx;
  int ny     = param.Ny;
  int Nt     = param.Nt;

  double eps = traj/Ns;
  double t = 0.;

  double* p = (double*)malloc(volume*sizeof(double));
  for (int i = 0; i < volume; i++) p[i] = BoxMuller();

  double Hi = Hamiltonian(phi, p, lattice, param);

  double* newphi = (double*)malloc(volume*sizeof(double));
  for (int i = 0; i < volume; i++) newphi[i] = phi[i];

  while (t < traj)
  {
    for (int i = 0; i < volume; i++)
      newphi[i] += 0.5 * eps * p[i];

    for (int i = 0; i < volume; i++)
      p[i] += eps * force(i, newphi,lattice,param);

    for (int i = 0; i < volume; i++)
      newphi[i] += 0.5 * eps * p[i];

    t += eps;
  }

  double Hf = Hamiltonian(newphi, p, lattice, param);

  int acc = 0;

  if (exp(-(Hf-Hi)) > drand48())
  {
    for (int i = 0; i < volume; i++) phi[i] = newphi[i];
    acc = 1;
  }

  cout << "delta H : " << Hf-Hi << endl << endl;

  free(p);
  free(newphi);

  return acc;
}


int main(int argc, char** argv)
{
  int L          = atoi(argv[1]);  // 8
  double kappa   = atof(argv[2]);  // 0.15
  double lambda  = atof(argv[3]);  // 1.1698
  int Nupdates   = atoi(argv[4]);  // 20000
  double trajlen = atof(argv[5]);  // 1.0
  int Ns         = atoi(argv[6]);  // 10

  int Nx = L;
  int Ny = L;
  int Nt = L;
  int volume = Nx*Ny*Nt;

  params param;
  param.Nx = Nx;
  param.Ny = Ny;
  param.Nt = Nt;
  param.kappa  = kappa;
  param.lambda = lambda;
  param.volume = volume;

  double* phi = (double*)malloc(volume*sizeof(double));
  for (int i = 0; i < volume; i++)
    phi[i] = 0. * BoxMuller();  // cold sart

  // assigning the coordinates of the sites
  site* lattice = (site*)malloc(volume*sizeof(site));
  for (int i = 0; i < volume; i++)
  {
    lattice[i].index = i;
    lattice[i].x = posx(i, Nx);
    lattice[i].y = posy(i, Nx, Ny);
    lattice[i].t = post(i, Nx, Ny);
  }

  // thermalization
  double leap_frog;
  for (int update = 0; update < 10000; update++)
    leap_frog = leapfrog_update(phi, lattice, param, trajlen, Ns);

  double acc_rate = 0.;
  double mAbsSum  = 0.;
  double m        = 0.;

  double* avphi   = (double*)malloc(Nt*sizeof(double));
  double* cor     = (double*)malloc(Nt*sizeof(double));
  double* avcor   = (double*)malloc(Nt*sizeof(double));
  double* avcorsq = (double*)malloc(Nt*sizeof(double));

  // for saving phi[0]
  FILE* phidat;
  phidat = fopen("phi.dat", "w");

  // HMC updates
  for (int update = 0; update < Nupdates; update++)
  {
    for (int i = 0; i < Nt; i++) { avphi[i] = 0; cor[i] = 0; }; m = 0.;

    cout << "update : " << update << endl;
    acc_rate += leapfrog_update(phi,lattice,param,trajlen,Ns)/(double)Nupdates;

    fprintf(phidat, "%f\n", phi[0]);
    for (int i = 0; i < volume; i++) m += phi[i];
    mAbsSum += abs(m);

    // projecting to zero momentum
    for(int i=0; i<volume; ++i)
      avphi[lattice[i].t] += phi[i];

    // calculating the correlation function
    for(int i=0; i<Nt; ++i)
      for(int j=0; j<Nt; ++j)
        cor[i] += avphi[j]*avphi[(j+i)%Nt];

    for(int i=0; i<Nt; ++i)
    {
      avcor[i]   += cor[i]/Nt;
      avcorsq[i] += pow(cor[i]/Nt,2);
    }
  }

  // printing the final results
  cout << "acceptance rate : " << acc_rate << endl;
  cout << "<|m|>/V : " << mAbsSum/(Nupdates*volume) << endl;
  for(int i=0; i<Nt; ++i)
    printf("PROP: %02d %f %f\n", i, avcor[i]/Nupdates,
           sqrt((avcorsq[i]/Nupdates-pow(avcor[i]/Nupdates,2))/Nupdates));

  free(phi);
  free(lattice);
  fclose(phidat);
  free(avphi);
  free(cor);
  free(avcor);
  free(avcorsq);

  return 0;
}
