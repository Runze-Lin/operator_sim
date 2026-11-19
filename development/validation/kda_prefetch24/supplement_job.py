"""Start the sealed supplementary queue with its own report module loaded first.

Legacy qualification also adds a directory with a report.py to sys.path. The
statistical publication step here uses the FwdH report from this directory.
"""
import sys

import report
from supplement import run
from supplement_score import publish


if __name__=='__main__':
    {'run':run,'publish':publish}[sys.argv[1]]()
