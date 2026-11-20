"""Use canonical module imports alongside the installation's legacy validation modules."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
import report
import documentation
from development.validation.quant29.collect import collect

if __name__=='__main__':
    collect()
