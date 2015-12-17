#!/usr/bin/env python
from __future__ import print_function, division
from collections import namedtuple
from mpi4py import MPI
from glob import glob
import numpy as np
import healpy as hp
import struct
import time
import os
import sys

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
    def __init__(self,fname,dtype,nmax=8000000):
        self.buff = np.zeros(nmax,dtype=dtype)
        self.nmax = nmax
        self.ncurr = 0
        self.fname = fname
        self.dumpcount = 0
        self.nwritten = 0

    def __del__(self):
        self.write()
        
    def write(self):
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
            
        if loc < len(d):
            nleft = len(d) - loc
            self.buff[self.ncurr:self.ncurr+nleft] = d[loc:loc+nleft]
            self.ncurr += nleft


class RBuffer(object):
    def __init__(self,fname,header,nmax=8000000,filenside=16):
        self.pbuff = np.zeros(nmax*3,dtype='f4')
        self.vbuff = np.zeros(nmax*3,dtype='f4')
        self.ibuff = np.zeros(nmax,dtype='u8')
        self.pidx = np.zeros(12*filenside**2,dtype='i8')

        self.ncurr = 0
        self.dumpcount = 0
        self.nwritten = 0
        self.fname = fname
        self.nmax = nmax
        self.filenside = header[2]
        self.fileorder = int(np.log2(self.filenside))
        self.header = header
        self.hdrfmt = 'fQIdddd'

    def sort_by_peano(self):
        pix = hp.vec2pix(self.filenside, self.pbuff[:3*self.ncurr:3], \
                         self.pbuff[1:3*self.ncurr:3], self.pbuff[2:3*self.ncurr:3], nest=True)
        peano = nest2peano(pix, self.fileorder)
        pidx = np.argsort(peano)
        pix = pix[pidx]
        for i in range(3):
            self.pbuff[i:3*self.ncurr:3] = self.pbuff[i:3*self.ncurr:3][pidx]
            self.vbuff[i:3*self.ncurr:3] = self.vbuff[i:3*self.ncurr:3][pidx]
            
        self.ibuff[:self.ncurr] = self.ibuff[pidx]

        pidx = pix[1:]-pix[:-1]
        pidx = np.where(pidx!=0)[0]+1
        nidx = [0]
        nidx.extend(list(pidx))
        pidx = np.array(nidx, dtype='i8')
        nparts = np.hstack([pidx[1:]-pidx[:-1], np.array([len(pix)-pidx[-1]])])
        self.pidx[pix[pidx]] = nparts
        self.header[1] = len(pix)

    def write(self):
        self.sort_by_peano()

        if self.ncurr > 0:
            #tprint("    writing to file '%s'" % self.fname+'.{0}'.format(self.dumpcount))
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
        while nnew + self.ncurr > self.nmax:
            nadd = self.nmax - self.ncurr
            
            self.pbuff[3*self.ncurr:3*(self.ncurr+nadd)] = pos[3*loc:3*(loc+nadd)]
            self.vbuff[3*self.ncurr:3*(self.ncurr+nadd)] = vel[3*loc:3*(loc+nadd)]
            self.ibuff[self.ncurr:self.ncurr+nadd] = ids[loc:loc+nadd]

            self.ncurr += nadd
            loc += nadd
            nnew -= nadd
            
            self.write()
            
        if loc < len(ids):
            nleft = len(ids) - loc
            self.pbuff[3*self.ncurr:3*(self.ncurr+nleft)] = pos[3*loc:3*(loc+nleft)]
            self.vbuff[3*self.ncurr:3*(self.ncurr+nleft)] = vel[3*loc:3*(loc+nleft)]
            self.ibuff[self.ncurr:self.ncurr+nleft] = ids[loc:loc+nleft]
            self.ncurr += nleft


def write_to_redshift_cells_buff(filepaths, outbase, cosmology, filenside=16, buffersize=1000000, 
                                 rmin=0, rmax=4000, rstep=25):
    """
    Read in gadget particle block, and write to the correct healpix/redshift
    cell files
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    rbins = np.linspace(rmin,rmax,(rmax-rmin)//rstep+1)
    rbins2 = rbins*rbins
    
    buffs = {}
    header = [1050.0, 0, filenside, 3.16, cosmology[0], cosmology[1], cosmology[2]]
        
    nfiles = len(filepaths)
    
    
    for fnum,filepath in enumerate(filepaths):
        tprint('    file %6d of %6d' % (fnum+1,nfiles))
        block = filepath.split('/')[-1].split('.')[-1]
        hdr, pos, vel, ids = readGadgetSnapshot(filepath,
                                                read_pos=True,
                                                read_vel=True,
                                                read_id=True,
                                                lgadget=True)
        tprint('    read data')
        
        r2 = pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2
        bins = np.ndarray(len(pos), dtype=np.int32)
        
        #Index by radial bin
        bins = np.digitize(r2, rbins2)
        
        tprint('    made indexes')
        
        #sort indices
        idx = bins.argsort()
        
        pos = pos[idx,:]
        vel = vel[idx,:]
        ids = ids[idx]
        bins = bins[idx]
        del idx
        
        tprint('    sorted data')
        
        #create index into radial cells
        rinc = bins[1:]-bins[:-1]
        
        idx, = np.where(rinc!=0)
        idx = list(idx+1)
        nidx = [0]
        nidx.extend(idx)
        idx = np.array(nidx,dtype='i8')
        
        #Write to disk
        nwrit = 0
        for i, start in enumerate(idx):
            if i==(len(idx)-1):
                end = len(bins)
            else:
                end = idx[i+1]

            rind = bins[start]
            deltan = end - start
            tprint('    Number of particles in bin {0}: {1}'.format(rind, deltan))
            nwrit += deltan
            
            if rind not in buffs:
                buffs[rind] = RBuffer(outbase+'_{0}_{1}'.format(bins[start], block), header,
                                      nmax=buffersize)
            buffs[rind].add(pos[start:end,:].flatten(), vel[start:end,:].flatten(),
                            ids[start:end])
            

        assert nwrit == len(bins)
        tprint('    put data in buff')
        
    tprint('    writing outputs')
    nwrit = 0
    for rind in buffs.keys():
        tprint('    bin %03d of %03d' % (rind,len(rbins)))
        buffs[rind].write()
        nwrit += buffs[rind].nwritten
        tprint('    Buffer for bin %03d of %03d dumped %03d times' % (rind,len(rbins),buffs[rind].dumpcount))
        del buffs[rind]
    tprint('    Total number of particles written, read: {0}, {1}'.format(nwrit, len(bins)))

    return rbins

def combine_radial_buffer_pair(file1, file2):

    b1 = file1.split('.')[-1]
    b2 = file2.split('.')[-1]
    ws = file1.split('.')[:-1]
    ws.append('join'+b1+b2)
    wf = '.'.join(ws)
    hdrfmt = 'fQIdddd'

    with open(file1, 'rb') as rp1:
        with open(file2, 'rb') as rp2:
            h1, idx1 = read_radial_bin(rp1)
            h2, idx2 = read_radial_bin(rp2)
            h = h1
            h[1] += h2[1]
            idx = idx1+idx2
            nadd = 0

            assert(np.sum(idx) == h[1])
            with open(wf, 'wb') as wp:
                wp.write(struct.pack(hdrfmt, *h))
                wp.write(idx.tobytes())

            #write particles
            buff = Buffer(wf, dtype='f4')
            fmt = np.dtype(np.float32)
            for i in range(len(idx1)):
                if idx1[i]:
                    d = np.fromstring(rp1.read(int(idx1[i]*3*fmt.itemsize)),fmt)
                    buff.add(d)
                    nadd += len(d)//3
                if idx2[i]:
                    d = np.fromstring(rp2.read(int(idx2[i]*3*fmt.itemsize)),fmt)
                    buff.add(d)
                    nadd += len(d)//3

            assert(nadd == h[1])
            buff.write()

            #write velocities
            for i in range(len(idx1)):
                if idx1[i]:
                    d = np.fromstring(rp1.read(int(idx1[i]*3*fmt.itemsize)),fmt)
                    buff.add(d)
                if idx2[i]:
                    d = np.fromstring(rp2.read(int(idx2[i]*3*fmt.itemsize)),fmt)
                    buff.add(d)

            buff.write()

            assert(buff.nwritten//6 == h[1])
            #write ids
            buff = Buffer(wf, dtype='u8')
            fmt = np.dtype(np.uint64)
            for i in range(len(idx1)):
                if idx1[i]:
                    d = np.fromstring(rp1.read(int(idx1[i]*fmt.itemsize)),fmt)
                    buff.add(d)
                if idx2[i]:
                    d = np.fromstring(rp2.read(int(idx2[i]*fmt.itemsize)),fmt)
                    buff.add(d)
            
            buff.write()

    return wf

def process_radial_cell(basepath, rbin, filenside=16):

    files = glob('{0}_{1}*'.format(basepath, rbin))
    processed = []

    while len(files)>1:
        files.append(combine_radial_buffer_pair(files[0], files[1]))
        processed.append(files[0])
        processed.append(files[1])
        files.remove(files[0])
        files.remove(files[0])

    os.rename(files[0], '{0}_{1}'.format(basepath, rbin))

    for f in processed:
        os.remove(f)


def read_radial_bin(filename, filenside=16, read_pos=False, \
                        read_vel=False, read_ids=False):

    hdrfmt = 'fQIdddd'
    idxfmt = np.dtype('i8')
    to_read = [read_pos, read_vel, read_ids]
    fmt = [np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.uint64)]
    item_per_row = [3,3,1]
    data = []
    filenpix  = 12*filenside**2
    opened = False
    if not hasattr(filename, 'read'):
        opened = True
        fp = open(filename, 'rb')
    else:
        fp = filename

    #read the header
    h = list(struct.unpack(hdrfmt, \
            fp.read(struct.calcsize(hdrfmt))))

    npart = h[1]
    data.append(h)
    #read the peano index
    idx = np.fromstring(fp.read(idxfmt.itemsize*filenpix), idxfmt)
    data.append(idx)

    for i, r in enumerate(to_read):
        if r:
            data.append(np.fromstring(fp.read(int(npart*item_per_row[i]*fmt[i].itemsize)), fmt[i]))

    if opened:
        fp.close()

    return data


def nest2peano(pix, order):
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

def map_LC_to_radial_bins(namefile, outpath, cosmology, rmin, rmax):

    with open(namefile, 'r') as fp:
        blockpaths = np.loadtxt(namefile)

    simlabel = blockpaths[0].split('/')[-1].split('_')[:3]
    outbase = '{0}/{1}'.format(outpath, simlabel)

    rbins = write_to_redshift_cells_buff(blockpaths, outbase, cosmology, 
                                         rmin=rmin, rmax=rmax)

def process_all_radial_bins(outbase, rmin, rmax, rstep=25.0):
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    rbins = np.arange((rmax-rmin)//rstep)
    chunks = [rbins[i::size] for i in range(size)]
    
    for r in chuncks[rank]:
        process_radial_cell(basepath, r)


if __name__ == '__main__':
    
    filelist = sys.argv[1]
    rmin = sys.argv[2]
    rmax 
