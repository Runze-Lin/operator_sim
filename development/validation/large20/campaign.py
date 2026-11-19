"""Run the registered paired Cast acquisition."""
import sys
from large_cast import collect, freeze

if __name__=="__main__":
    {"freeze":freeze,"collect":collect}[sys.argv[1]]()
