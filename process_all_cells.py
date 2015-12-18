import pixLC as pl
import sys


if __name__ == '__main__':

    outbase = sys.argv[1]
    rmin = float(sys.argv[2])
    rmax = float(sys.argv[3])

    pl.process_all_cells(outbase, rmin, rmax, rstep=3)
