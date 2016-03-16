# pixLC
Reformat a particle lightcone into radial bins and healpix cells. 
Uses a fixed radial bin width of 25 Mpc/h, and breaks each bin
into healpix cells of increasing resolution to maintain approximately
the same file size. All healpix cells are numbered according to the NESTED
ordering scheme.

Files are not created for cells with no particles in them. The exception
to this is that radial bins are guaranteed to always have a file for 
healpix cell 0. This file may be used to determine the nside used for 
its radial bin.

Header format:

    unsigned long npart      : number of particles in the present file
    unsigned int indexnside       : nside value used to sort particles within this file
    unsigned int filenside   : nside used to break up radial bin this file falls in
    float BoxSize		     : in Mpc/h
    double mass              : particle mass in 1e10 M_sun/h
    unsigned long npartTotal : total number of particles in the box
    double Omega0           
    double OmegaLambda      
    double HubbleParam       : little 'h'

    
The particles are sorted by the Peano-Hilbert index of the healpix cell they fall within using an nside value specified by the indexnside field in the header of the file.
Immediately after the header there is a list of $12\times indexnside^2$ unsigned long ints corresponding to the number of particles contained in each healpix cell of nside=indexnside. The healpix cells in this index are sorted according to their Peano-Hilbers index as well.

Following this index is the particle data formatted as follows:
    
    positions  : 3*npart floats
    velocities : 3*npart floats
    ids        : 1*npart unsigned long int
 

    
  
