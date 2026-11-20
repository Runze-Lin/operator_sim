"""Pin the report for this prospective study before old collectors extend sys.path."""
import sys
from pathlib import Path
# Older collection helpers have a generic module named report.
from report import write_report
from channel import run
if __name__=='__main__':run()
