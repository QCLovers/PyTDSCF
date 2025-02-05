import pytest
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except ImportError | ModuleNotFoundError:
    RANK = 0

def pytest_configure(config):
    if RANK != 0:
        config.option.capture = 'no'
        config.option.verbose = 0
        import os
        import sys
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if RANK == 0:
        yield
    else:
        outcome = yield
        terminalreporter.stats.clear()

def pytest_collection_modifyitems(session, config, items):
    if RANK != 0:
        items.clear()

def pytest_sessionfinish(session, exitstatus):
    if RANK != 0:
        session.exitstatus = 0  # Always success except rank0
