# pixLC
Reformat a particle lightcone into radial bins and healpix cells. 
Uses a fixed radial bin width of 25 Mpc/h, and breaks each bin
into healpix cells of increasing resolution to maintain approximately
the same file size.

Files are not created for cells with no particles in them. The exception
to this is that radial bins are guaranteed to always have a file for 
healpix cell 0. This file may be used to determine the nside used for 
its radial bin.

Header format:

  unsigned long npart      // number of particles in the present file
  unsigned int nside       // nside used to break up radial bin this file falls in
  unsigned int filenside   // nside value used to sort particles within this file
  float BoxSize		   // in Mpc/h
  double mass          	   // particle mass in 1e10 M_sun/h
  unsigned long npartTotal // total number of particles in the box
  double Omega0;           
  double OmegaLambda;      
  double HubbleParam;      // little 'h'
