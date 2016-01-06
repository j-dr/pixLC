import pixLC as pl
import sys

if __name__ == '__main__':

    outbase = sys.argv[1]
    rmin = float(sys.argv[2])
    rmax = float(sys.argv[3])
    lfilenside = int(sys.argv[4])
    rr0 = float(sys.argv[5])

    pl.process_all_cells(outbase, rmin, rmax, rstep=25.0, rr0=rr0, lfilenside=lfilenside, hfilenside=4)
