#!/usr/bin/env python
from __future__ import print_function, division
from collections import namedtuple, deque
from mpi4py import MPI
from glob import glob
from copy import copy
import numpy as np
import healpy as hp
import struct
import time
import os
import sys
import yaml

TZERO = None
def tprint(info):
    global TZERO
    if TZERO is None:
        TZERO = time.time()

    print('[%8ds] %s' % (time.time()-TZERO,info))

__GadgetHeader_fmt = '6I6dddii6Iiiddddii6Ii'
__finenside = 8192

GadgetHeader = namedtuple('GadgetHeader', \
        'npart mass time redshift flag_sfr flag_feedback npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda HubbleParam flag_age flag_metals NallHW flag_entr_ics')

def readGadgetSnapshot(filename, read_pos=False, read_vel=False, read_id=False,\
        read_mass=False, print_header=False, single_type=-1, lgadget=False):
    """
    This function reads the Gadget-2 snapshot file.

    Parameters
    ----------
    filename : str
        path to the input file
    read_pos : bool, optional
        Whether to read the positions or not. Default is false.
    read_vel : bool, optional
        Whether to read the velocities or not. Default is false.
    read_id : bool, optional
        Whether to read the particle IDs or not. Default is false.
    read_mass : bool, optional
        Whether to read the masses or not. Default is false.
    print_header : bool, optional
        Whether to print out the header or not. Default is false.
    single_type : int, optional
        Set to -1 (default) to read in all particle types. 
        Set to 0--5 to read in only the corresponding particle type.
    lgadget : bool, optional
        Set to True if the particle file comes from l-gadget. 
        Default is false.

    Returns
    -------
    ret : tuple
        A tuple of the requested data. 
        The first item in the returned tuple is always the header.
        The header is in the GadgetHeader namedtuple format.
    """
    blocks_to_read = (read_pos, read_vel, read_id, read_mass)
    ret = []
    with open(filename, 'rb') as f:
        f.seek(4, 1)
        h = list(struct.unpack(__GadgetHeader_fmt, \
                f.read(struct.calcsize(__GadgetHeader_fmt))))
        if lgadget:
            h[30] = 0
            h[31] = h[18]
            h[18] = 0
            single_type = 1
        h = tuple(h)
        header = GadgetHeader._make((h[0:6],) + (h[6:12],) + h[12:16] \
                + (h[16:22],) + h[22:30] + (h[30:36],) + h[36:])
        if print_header:
            print( header )
        if not any(blocks_to_read):
            return header
        ret.append(header)
        f.seek(256 - struct.calcsize(__GadgetHeader_fmt), 1)
        f.seek(4, 1)
        #
        mass_npart = [0 if m else n for m, n in zip(header.mass, header.npart)]
        if single_type not in range(6):
            single_type = -1
        #
        for i, b in enumerate(blocks_to_read):
            if i < 2:
                fmt = np.dtype(np.float32)
                item_per_part = 3
                npart = header.npart
            elif i==2:
                fmt = np.dtype(np.uint64) if lgadget or any(header.NallHW) \
                        else np.dtype(np.uint32)
                item_per_part = 1
                npart = header.npart
            elif i==3:
                fmt = np.dtype(np.float32)
                if sum(mass_npart) == 0:
                    ret.append(np.array([], fmt))
                    break
                item_per_part = 1
                npart = mass_npart
            size_per_part = item_per_part*fmt.itemsize
            #
            f.seek(4, 1)
            if not b:
                f.seek(sum(npart)*size_per_part, 1)
            else:
                if single_type > -1:
                    f.seek(sum(npart[:single_type])*size_per_part, 1)
                    npart_this = npart[single_type]
                else:
                    npart_this = sum(npart)
                data = np.fromstring(f.read(npart_this*size_per_part), fmt)
                if item_per_part > 1:
                    data.shape = (npart_this, item_per_part)
                ret.append(data)
                if not any(blocks_to_read[i+1:]):
                    break
                if single_type > -1:
                    f.seek(sum(npart[single_type+1:])*size_per_part, 1)
            f.seek(4, 1)
    #
    return tuple(ret)

        
class Buffer(object):
    """
    Buffer object for use when recombining cells written in map step.
    Handles one type of data (pos, vel, ids) at a time.
    """
    def __init__(self,fname,dtype,nmax=10000000):
        """
        Initialize buffer
        
        fname -- The filename to write to
        dtype -- The type of data to write
        nmax -- The max number of particles to hold at once before dumping
        """
        self.buff = np.zeros(nmax,dtype=dtype)
        self.nmax = nmax
        self.ncurr = 0
        self.fname = fname
        self.dumpcount = 0
        self.nwritten = 0

    def __del__(self):
        self.write()
        
    def write(self):
        #if buffer not empty, write 
        print('ncurr: {0}'.format(self.ncurr))
        if self.ncurr > 0:
            with open(self.fname, 'ab') as fp:
                fp.write(self.buff[0:self.ncurr].tobytes())

            self.nwritten += self.ncurr
            self.ncurr = 0
            self.dumpcount += 1

    def add(self,d):
        assert d.dtype == self.buff.dtype
        
        loc = 0
        nnew = len(d)

        #if more particles added than can be 
        #held in buffer, dump
        while nnew + self.ncurr > self.nmax:
            npos = self.nmax - self.ncurr
            if npos > nnew:
                nadd = nnew
            else:
                nadd = npos
            
            self.buff[self.ncurr:self.ncurr+nadd] = d[loc:loc+nadd]
            self.ncurr += nadd
            loc += nadd
            nnew -= nadd
            
            self.write()

        #if all particles not yet in buffer, put the rest in
        if loc < len(d):
            nleft = len(d) - loc
            self.buff[self.ncurr:self.ncurr+nleft] = d[loc:loc+nleft]
            self.ncurr += nleft

class RBuffer(object):
    """
    Buffer object to use for map step. Holds position, velocity, and ids at once
    """

    def __init__(self,fname,header,nmax=8000000):
        """
        Initialize buffer
        
        fname -- File name to write to
        header -- File header to write
        nmax -- Max number of particles to hold at once
        """

        self.pbuff = np.zeros(nmax*3,dtype='f4')
        self.vbuff = np.zeros(nmax*3,dtype='f4')
        self.ibuff = np.zeros(nmax,dtype='u8')
        self.pidx = np.zeros(12*header[1]**2,dtype='i8')

        self.ncurr = 0
        self.dumpcount = 0
        self.nwritten = 0
        self.fname = fname
        self.nmax = nmax
        self.indexnside = header[1]
        self.fileorder = int(np.log2(self.indexnside))
        self.header = header
        self.hdrfmt = 'QIIfdQddd' 

    def sort_by_peano(self):
        """
        Sort particles by the peano index of the healpix cell
        they are in
        """
        #determine healpix cells
        pix = hp.vec2pix(self.indexnside, self.pbuff[:3*self.ncurr:3], \
                         self.pbuff[1:3*self.ncurr:3], self.pbuff[2:3*self.ncurr:3], nest=True)
        peano = nest2peano(pix, self.fileorder)
        #sort by peano index
        pidx = np.argsort(peano)
        peano = peano[pidx]
        for i in range(3):
            self.pbuff[i:3*self.ncurr:3] = self.pbuff[i:3*self.ncurr:3][pidx]
            self.vbuff[i:3*self.ncurr:3] = self.vbuff[i:3*self.ncurr:3][pidx]
            
        self.ibuff[:self.ncurr] = self.ibuff[pidx]

        #determine number of particles in each cell
        pidx = peano[1:]-peano[:-1]
        pidx = np.where(pidx!=0)[0]+1
        nidx = [0]
        nidx.extend(list(pidx))
        pidx = np.array(nidx, dtype='i8')
        nparts = np.hstack([pidx[1:]-pidx[:-1], np.array([len(peano)-pidx[-1]])])
        self.pidx[peano[pidx]] = nparts
        self.header[0] = len(peano)

    def write(self):
        self.sort_by_peano()
        
        #if buffer not empty, write data to file
        if self.ncurr > 0:
            with open(self.fname+'.{0}'.format(self.dumpcount), 'w+b') as fp:
                fp.write(struct.pack(self.hdrfmt, *self.header))
                fp.write(self.pidx.tobytes())
                fp.write(self.pbuff[0:3*self.ncurr].tobytes())
                fp.write(self.vbuff[0:3*self.ncurr].tobytes())
                fp.write(self.ibuff[0:self.ncurr].tobytes())
            
            self.nwritten += self.ncurr
            self.ncurr = 0
            self.pidx[:] = 0
            self.dumpcount += 1

    def add(self, pos, vel, ids):
        assert pos.dtype == self.pbuff.dtype
        assert vel.dtype == self.vbuff.dtype
        assert ids.dtype == self.ibuff.dtype
        
        loc = 0
        nnew = len(ids)
        assert nnew*3==len(pos)
        
        #if number of particles added greater than
        #buffer capacity, dump
        while nnew + self.ncurr > self.nmax:
            nadd = self.nmax - self.ncurr
            
            self.pbuff[3*self.ncurr:3*(self.ncurr+nadd)] = pos[3*loc:3*(loc+nadd)]
            self.vbuff[3*self.ncurr:3*(self.ncurr+nadd)] = vel[3*loc:3*(loc+nadd)]
            self.ibuff[self.ncurr:self.ncurr+nadd] = ids[loc:loc+nadd]

            self.ncurr += nadd
            loc += nadd
            nnew -= nadd
            
            self.write()
            
        #if haven't stored all particles added, do so
        if loc < len(ids):
            nleft = len(ids) - loc
            self.pbuff[3*self.ncurr:3*(self.ncurr+nleft)] = pos[3*loc:3*(loc+nleft)]
            self.vbuff[3*self.ncurr:3*(self.ncurr+nleft)] = vel[3*loc:3*(loc+nleft)]
            self.ibuff[self.ncurr:self.ncurr+nleft] = ids[loc:loc+nleft]
            self.ncurr += nleft

def create_refinement_plan(rmin, rmax, rstep, rr0, lfilenside, hfilenside=None):
    """
    Determine the nside to write files to for each radial bin
    
    rmin -- Minimum radius
    rmax -- Maximum radius
    rstep -- Radial bin size
    rr0 -- The first radius to refine at
    lfilenside -- Nside to use at rmin
    hfilenside -- Max nside to use
    """

    rbins = np.linspace(rmin,rmax,(rmax-rmin)//rstep+1)

    #determine refinement radii
    rr = [rr0]
    while rr[-1]<rmax:
        rr.append(rr[-1]*np.sqrt(2))
        
    #determine nside of each radial bin
    rnside = np.zeros(len(rbins)-1, dtype=np.int32)
    rnside[0] = lfilenside
    nr = 0
    for i, r in enumerate(rbins[1:-1]):
        #if lower edge of radial bin greater than next refinement
        #radius. refine the grid
        if r>rr[nr]: nr+=1
        rnside[i+1] = lfilenside*2**nr
        if (hfilenside!=None) & (rnside[i+1]>hfilenside):
            rnside[i+1] = hfilenside

    return rbins, rr, rnside, nr



def write_to_cells_buff(filepaths, outbase, indexnside=16, lfilenside=1, 
                        hfilenside=None, rr0=300.0, buffersize=1000000, rmin=0, 
                        rmax=4000, rstep=25, boxsize=1050, pmass=3.16):
    """
    Read in gadget particle block, and write to the correct healpix/redshift
    cell files. 
    
    parameters:
    lfilenside: int
        The healpix nside to start with
    hfilenside: int
        The highest nside that we want to use
    rr0: float
        The radius at which we want to perform our first refinement. Further refinements
        will be performed at r_n = rr0*sqrt(2)^n.
    """
    #determine the radii to refine the nside value of the healpix cells that
    #the files are broken into
    assert((rmin%rstep==0) and (rmax%rstep==0))
    rbins, rr, rnside, nr = create_refinement_plan(rmin, rmax, rstep, rr0,
                                                   lfilenside, hfilenside=hfilenside)
    bin_offset = rmin//rstep
    rbins2 = rbins*rbins

    print('Max number of refinements: {0}'.format(nr))
    print('Maximum nside: {0}'.format(rnside[-1]))
    print('Radii to refine at: {0}'.format(rr))
    
    header = [0, indexnside, rnside[0], 0.0, 0.0, 0, 0.0, 0.0, 0.0]
    ntot = 0
        
    nfiles = len(filepaths)
    buffs = {}
    
    #iterate over files, writing them to radial/healpix cells
    for fnum,filepath in enumerate(filepaths):
        tprint('    file %6d of %6d' % (fnum+1,nfiles))
        block = filepath.split('/')[-1].split('.')[-1]
        hdr, pos, vel, ids = readGadgetSnapshot(filepath,
                                                read_pos=True,
                                                read_vel=True,
                                                read_id=True,
                                                lgadget=True)
        header[-1] = hdr.HubbleParam
        header[-2] = hdr.OmegaLambda
        header[-3] = hdr.Omega0
        header[-4] = hdr.npartTotal[1]
        header[-5] = hdr.mass[1]
        header[-6] = hdr.BoxSize
        
        r2 = pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2

        #discard partices that don't fall into the desired radial range
        ridx = (rmin**2<=r2) & (r2<rmax**2)
        r2 = r2[ridx]
        if len(r2)==0:
            continue

        pos = pos[ridx,:]
        vel = vel[ridx,:]
        ids = ids[ridx]
        
        #determine the healpix cell of parts using largest radial bin's nside 
        pix = hp.vec2pix(rnside[-1], pos[:,0], pos[:,1], pos[:,2], nest=True)
        idxdtype = np.dtype([('pidx', np.int32), ('ridx', np.int32)])
        bins = np.ndarray(len(pos), dtype=idxdtype)

        #Index by radial bin and highest order hpix cell
        bins['ridx'] = np.digitize(r2, rbins2)-1
        bins['pidx'] = np.digitize(pix, np.arange(12*rnside[-1]**2))

        #sort indices
        idx = bins.argsort(order=['ridx','pidx'])
        pos = pos[idx,:]
        vel = vel[idx,:]
        ids = ids[idx]
        bins = bins[idx]
        del idx, pix
        
        #create index into radial cells by
        #finding where radial bin increments
        rinc = bins['ridx'][1:]-bins['ridx'][:-1]
        idx, = np.where(rinc!=0)
        idx = list(idx+1)
        nidx = [0]
        nidx.extend(idx)
        idx = np.array(nidx,dtype='i8')
        
        #Write to disk
        nwrit = 0
        for i, start in enumerate(idx):
            header = copy(header)
            if i==(len(idx)-1):
                end = len(bins)
            else:
                end = idx[i+1]
            
            rind = bins['ridx'][start]
            header[2] = rnside[rind]

            print('Working on radial bin with nside = {0}'.format(rnside[rind]))
            #Determine hpix cell of parts with nside for this radial bin
            pix = hp.vec2pix(rnside[rind], pos[start:end,0], pos[start:end,1],
                             pos[start:end,2], nest=True)
            
            #Create index into hpix values, already sorted since sorted by
            #largest radial bin's nside earlier
            pinc = pix[1:]-pix[:-1]
            pidx, = np.where(pinc!=0)
            pidx = list(pidx+1)
            nidx = [0]
            nidx.extend(pidx)
            pidx = np.array(nidx,dtype='i8')
            
            #iterate over hpix and write parts to cells
            for j, pstart in enumerate(pidx):
                if j==(len(pidx)-1):
                    pend = len(pix)
                else:
                    pend = pidx[j+1]
                    
                pind = pix[pstart]
                deltan = pend - pstart

                nwrit += deltan
                if rind not in buffs:
                    buffs[rind] = {}
                if pind not in buffs[rind]:
                    buffs[rind][pind] = RBuffer(outbase+'_{0}_{1}_{2}'.format(int(rind+bin_offset), pind, block),
                                                header, nmax=buffersize)
                    
                buffs[rind][pind].add(pos[start+pstart:start+pend,:].flatten(),
                                      vel[start+pstart:start+pend,:].flatten(),
                                      ids[start+pstart:start+pend])
        
        assert nwrit == len(bins)
        ntot += len(bins)
        
    nwrit = 0
    #write out particles left in buffers, make sure we wrote everything
    for rind in buffs.keys():
        for pind in buffs[rind].keys():
            buffs[rind][pind].write()
            nwrit += buffs[rind][pind].nwritten
            del buffs[rind][pind]

    assert( nwrit == ntot )

    return header


def combine_cell_list(flist):
    """
    Combines a list of buffer dumps for a particular radial/hpix 
    cell into a single file maintaining peano ordering of particles
    """

    #Create unique file name to write combined file to
    for i, f in enumerate(flist):
        if 'mg' in f:
            fs = f.split('mg')
            fs[-1] = str(int(fs[-1])+1)
            wf = 'mg'.join(fs)
            break
        if i==(len(flist)-1):
            wf = f+'mg1'

    #open the files
    hdrfmt = 'QIIfdQddd'
    rps = [open(f, 'rb') for f in flist]
    idxs = []

    #read header, peano indices of all files
    #combine into one header index by summing
    #number of particles
    for i, rp in enumerate(rps):
        hi, idxi = read_radial_bin(rp)
        if i==0:
            h = hi
            idx = copy(idxi)
        else:
            h[0] += hi[0]
            idx += idxi

        assert( hi[2:] == h[2:] )
        
        idxs.append(idxi)
        nadd = 0

    assert(np.sum(idx) == h[0])
    #write new header, index
    with open(wf, 'wb') as wp:
        wp.write(struct.pack(hdrfmt, *h))
        wp.write(idx.tobytes())

    #put positions in buffer
    buff = Buffer(wf, dtype='f4')
    fmt = np.dtype(np.float32)
    for i in range(len(idx)):
        for j, idxi in enumerate(idxs):
            if idxi[i]:
                d = np.fromstring(rps[j].read(int(idxi[i]*3*fmt.itemsize)),fmt)
                buff.add(d)
                nadd += len(d)//3

    buff.write()
    #put velocities in buffer
    for i in range(len(idx)):
        for j, idxi in enumerate(idxs):
            if idxi[i]:
                d = np.fromstring(rps[j].read(int(idxi[i]*3*fmt.itemsize)),fmt)
                buff.add(d)
                nadd += len(d)//3
                
    buff.write()
    #make sure buffer wrote everything we put in it

    assert(buff.nwritten//6 == h[0])

    #create new buffer for ids
    buff = Buffer(wf, dtype='u8')
    fmt = np.dtype(np.uint64)
    for i in range(len(idx)):
        for j, idxi in enumerate(idxs):
            if idxi[i]:
                d = np.fromstring(rps[j].read(int(idxi[i]*3*fmt.itemsize)),fmt)
                buff.add(d)
        
    buff.write()
    
    for rp in rps:
        rp.close()

    return wf


def write_empty_header(basepath, rbin, pix, header):
    """
    Write empty header containing the file and index nside.
    Only want to do this for pixel 0 of each radial bin if
    empty for addgals to reference.

    basepath -- Path to write file to
    rbin -- Index of radial bin
    pix -- Healpix cell value
    header -- The header containing the correct file nside, index
              nside, cosmology and other metadata
    """
    hdrfmt = 'QIIfdQddd'
    header[0] = 0
    with open('{0}_{1}_{2}'.format(basepath, rbin, pix), 'w') as fp:
        fp.write(struct.pack(hdrfmt, *header))


def process_cell(basepath, rbin, pix, rank=None, ncomb=10, header=None):
    """
    Collect combine all buffer dumps for a particular radial/hpix cell into
    a single file.
    
    basepath -- Path to write the files to
    rbin -- Index of the radial bin
    pix -- Healpix cell value
    rank -- The rank of the mpi process assigned to this cell
    ncomb -- Number of buffers to combine at once
    header -- Header containing lightcone metadata for this cell
    """
    files = deque(glob('{0}_{1}_{2}_*'.format(basepath, rbin, pix)))
    if len(files)==0:
        write_empty_header(basepath, rbin, pix, header)
        return

    tprint("    {2} Processing cell {0} {1}, nfiles = {3}".format(rbin, pix, rank, len(files)))
    processed = []

    #iterating through files combining them ncomb at a time
    while len(files)>1:
        npop = min(ncomb, len(files))
        flist = [files.popleft() for i in range(npop)]
        files.append(combine_cell_list(flist))
        processed.extend(flist)

    os.rename(files[0], '{0}_{1}_{2}'.format(basepath, rbin, pix))

    for f in processed:
        os.remove(f)

    tprint("    {2} Done processing cell {0} {1}".format(rbin, pix, rank))


def read_radial_bin(filename, read_pos=False, read_vel=False, read_ids=False):
    """
    Read in a radial/hpix cell
    
    filename -- The name of the file to read, or a file object. If file
                object, will not be closed upon function return. Instead
                the pointer will be left at the location of the last 
                data read.
    read_xxx -- Whether or not to read xxx
    """
    hdrfmt = 'QIIfdQddd'
    idxfmt = np.dtype('i8')
    to_read = np.array([read_pos, read_vel, read_ids])
    fmt = [np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.uint64)]
    item_per_row = [3,3,1]
    data = []

    opened = False
    if not hasattr(filename, 'read'):
        opened = True
        fp = open(filename, 'rb')
    else:
        fp = filename

    #read the header
    h = list(struct.unpack(hdrfmt, \
            fp.read(struct.calcsize(hdrfmt))))

    npart = h[0]
    indexnside = h[1]
    indexnpix  = 12*indexnside**2
    data.append(h)
    #read the peano index
    idx = np.fromstring(fp.read(idxfmt.itemsize*indexnpix), idxfmt)
    data.append(idx)

    if to_read.any():
        for i, r in enumerate(to_read):
            d = np.fromstring(fp.read(int(npart*item_per_row[i]*fmt[i].itemsize)), fmt[i])
            if r:
                data.append(d)
            if not to_read[i+1:].any():break

        if opened:
            fp.close()

    return data


def nest2peano(pix, order):
    """
    Convert nest ordered healpix cell values to peano indices
    
    pix -- Healpix cell value
    order -- The order of the nside of pix
    """
    subpix = np.array([ [ 0, 1, 3, 2 ], [ 3, 0, 2, 1 ], [ 2, 3, 1, 0 ], [ 1, 2, 0, 3 ],\
                            [ 0, 3, 1, 2 ], [ 1, 0, 2, 3 ], [ 2, 1, 3, 0 ], [ 3, 2, 0, 1 ] ])
    subpath = np.array([ [ 4, 0, 6, 0 ], [ 7, 5, 1, 1 ], [ 2, 4, 2, 6 ], [ 3, 3, 7, 5 ],\
                             [ 0, 2, 4, 4 ], [ 5, 1, 5, 3 ], [ 6, 6, 0, 2 ], [ 1, 7, 3, 7 ] ])
    face2path = np.array([ 2, 5, 2, 5, 3, 6, 3, 6, 2, 3, 2, 3 ])
    face2peanoface = np.array([ 0, 5, 6, 11, 10, 1, 4, 7, 2, 3, 8, 9 ])
    
    npix_ = 12*(1 << (2*order))
    assert((pix >= 0).all() and (pix < npix_).all())
  
    face = pix>>(2*order)
    path = face2path[face]
    result = np.zeros(len(pix), dtype=np.int64);
    shifts = np.arange(0, 2*order-1, 2)
    
    for shift in shifts[::-1]:
        spix = (pix>>shift) & 0x3
        result <<= 2
        result |= subpix[path,spix]
        path = subpath[path,spix]
        
    return result + ((face2peanoface[face])<<(2*order));

def map_LC_to_cells(namefile, outpath, simlabel, rmin, rmax, lfilenside, rr0,
                    hfilenside=None):
    """
    Given a list of lightcone outputs from L-Gadget2, write the particles to radial/hpix cells 
    breaking up higher radial bins into more refined healpix cells.
    
    namefile -- Name of the file containing the list of L-Gadget2 snapshots
    outpath -- Path to write the reformatted files to
    simlabel -- Label of the snapshots, i.e. lightcone_Snapshot
    rmin -- Minimum radius to keep particles
    rmax -- Maxmimum radius to keep particles
    lfilenside -- Nside to write files to for rmin
    rr0 -- First radius to refine nside at
    hfilenside -- The highest nside to use
    """
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    blockpaths = np.genfromtxt(namefile, dtype=None)
    outbase = '{0}/{1}'.format(outpath, simlabel)

    step = (len(blockpaths) + size -1 ) // size
    chunks = [blockpaths[i*step:(i+1)*step] for i in range(size)]

    header = write_to_cells_buff(chunks[rank], outbase, lfilenside=lfilenside,
                                 hfilenside=hfilenside, rr0=rr0, rmin=rmin, rmax=rmax)
    return header

def process_all_cells(outbase, rmin, rmax, rstep=25.0, rr0=300.0, lfilenside=1,
                      hfilenside=None, header=None):
    """
    Combine the buffer dumps from map_LC_to_cells into single files for 
    each cell
    
    outbase -- Path to write files to
    rmin -- Minimum radius used
    rmax -- Maximum radius used
    rstep -- Radial bin size
    rr0 -- First radius to refine at 
    lfilenside -- Nside to use for rmin
    hfilenside -- Largest nside to use
    header -- Default header
    """
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    assert((rmin%rstep==0) and (rmax%rstep==0))
    #determine the nside to use for each radial bin
    rads, rr, rnside, nr = create_refinement_plan(rmin, rmax, rstep, rr0,
                                                   lfilenside, hfilenside=hfilenside)

    #if rmin is not zero add offset so that bin index is assigned unique radial range
    bin_offset = rmin//rstep
    rbins = np.arange((rmax-rmin)//rstep, dtype=np.int64)

    #determine number of pixels for each radial bin
    rnpix = 12*rnside**2
    idx = np.cumsum(rnpix)
    idx = np.hstack([np.zeros(1),idx])
    cells = np.ndarray((idx[-1],2), dtype=np.int64)

    for i, r in enumerate(rbins):
        cells[idx[i]:idx[i+1],0] = int(r+bin_offset)
        cells[idx[i]:idx[i+1],1] = np.arange(rnpix[i], dtype=np.int64)
    
    chunks = [cells[i::size,:] for i in range(size)]
    
    for i, c in enumerate(chunks[rank]):
        if i%50==0:
            tprint('    Worker {0} has processed {1}% of assigned cells'.format(rank, i/len(chunks[rank])))
        
        header[2] = rnside[int(c[0]-bin_offset)]
        process_cell(outbase, *c, rank=rank, header=header)


def readCFG(filename):
    
    with open(filename, 'r') as fp:
        pars = yaml.load(fp)

    return pars

if __name__=='__main__':

    comm = MPI.COMM_WORLD

    cfgfile = sys.argv[1]
    pars = readCFG(cfgfile)

    namefile = pars['namefile']
    outpath = pars['outpath']
    rmin = pars['rmin']
    rmax = pars['rmax']
    lfilenside = pars['lfilenside']
    rr0 = pars['rr0']
    prefix = pars['prefix']
    if 'process_only' in pars:
        process_only = pars['process_only']
    else:
        process_only = False

    try:
        os.makedirs(outpath)
    except OSError as e:
        pass

    outbase = '{0}/{1}'.format(outpath,prefix)

    if not process_only:
        header = map_LC_to_cells(namefile, outpath, prefix, rmin, rmax, lfilenside,
                                 rr0, hfilenside=4)
        comm.Barrier()
    else:
        pfiles = glob(outbase+'*')
        header, idx = read_radial_bin(pfiles[0])

    process_all_cells(outbase, rmin, rmax, rstep=25.0, rr0=rr0, lfilenside=lfilenside, hfilenside=4, header=header)
