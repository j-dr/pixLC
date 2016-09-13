#!/usr/bin/env python
from __future__ import print_function
from helpers import SimulationAnalysis
from fast3tree import fast3tree
from mpi4py import MPI

import numpy as np
import pixLC
import sys

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    cfg = sys.argv[1]
    halofile = sys.argv[2]

    cfg = pixLC.readCFG(cfg)

    inglob = "{0}/{1}_*".format(cfg['outpath'], cfg['prefix'])
    files  = glob(inglob)
    files = files[rank::size]

    halos = SimulationAnalysis.readHlist(halofile, ['x', 'y', 'z', 'mvir', 'rvir', 'pid'])

    halos = halos[halos['pid']==-1]
    hpos  = halos[['x', 'y', 'z']].view((h.dtype['x'],3)

    firstex = True
    with fast3tree(hpos) as tree:

        for f in files:
            #read in particle positions, reshape
            hdr, idx, ppos = pixLC.read_radial_bin(f, read_pos=True)
            ppos = pos.reshape(-1,3)
            hpd = np.zeros((len(ppos)),
                              dtype=np.dtype([('HALOID', np.int64),
                                              ('RHALO', np.float64),
                                              ('MVIR', np.float64),
                                              ('RVIR', np.float64)])) - 1
            for i, p in enumerate(ppos):

                try:
                    #get indices of all halos within radius
                    hsi = tree.query_radius(p, 15)
                    #get distances to those halos
                    di = np.sqrt((hpos[hsi,0]-p[0])**2 +
                        (hpos[hsi,1]-p[1])**2 + (hpos[hsi,2]-p[2])**2)
                    #find the closest
                    d = np.min(di)
                    hi = di==d

                    hpd['RHALO'][i] = d
                    hpd['HALOID'][i] = halos[hsi[hi]]['ID']
                    hpd['MVIR'][i] = halos[hsi[hi]]['MVIR']
                    hpd['RVIR'][i] = halos[hsi[hi]]['RVIR']

                #if fails, no halos within search radius
                except Exception as e:
                    if firstex:
                        print(e)
                    continue

            with open('hinfo_{0}'.format(f), 'w') as fp:
                fp.write(hpd.tobytes())
