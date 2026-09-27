"""Knowledge tracing from scratch: BKT, PFA, IRT and DKT on student answer logs."""

import os

# Every matrix here is small (at most 32 x 512). Multi-threaded BLAS spends more
# time synchronising than computing on those: measured on this machine, one DKT
# chunk took 1.28 s with the default thread pool and 0.15 s with one thread.
# setdefault, so an explicit setting by the caller still wins. This only takes
# effect if numpy has not been imported yet.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
