"""
Logging configuration for pytdscf
"""

import os
import sys

from loguru import logger


class Rank0Sink:
    """A custom sink that only writes on rank 0"""

    def __init__(self, sink):
        """
        Args:
            sink: The actual sink to write to (e.g. file or sys.stderr)
        """
        try:
            from mpi4py import MPI

            self.rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            self.rank = 0

        self.sink = sink

    def write(self, message):
        """Write message only if on rank 0"""
        if self.rank == 0:
            self.sink.write(message)

    def flush(self):
        """Flush only if on rank 0"""
        if self.rank == 0:
            self.sink.flush()

    @property
    def name(self):
        """Pass through the name attribute of the underlying sink"""
        return getattr(self.sink, "name", None)


def make_filter(name: str):
    """Create a filter for specific logger name

    Args:
        name: Name of the logger to filter for
    """

    def filter_(record):
        return record["extra"].get("name") == name

    return filter_


# デフォルトのmainロガーを設定
logger.remove()
logger.add(
    Rank0Sink(sys.stderr),
    format="{time:HH:mm:ss} | {level} | {message}",
    level="INFO",
    filter=make_filter("main"),
)


def setup_loggers(jobname: str, adaptive: bool = False):
    """Setup loguru loggers with rank 0 filtering and file outputs

    Args:
        jobname: Name of the job/directory for log files
    """
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except Exception:
        rank = 0
        size = 1

    if rank == 0 and not os.path.exists(f"./{jobname}"):
        os.makedirs(f"./{jobname}")

    if size > 1:
        comm.barrier()

    main_log_path = f"{jobname}/main.log"
    existing_handlers = [h for h in logger._core.handlers.values()]
    existing_paths = [
        h._sink.name if hasattr(h._sink, "name") else None
        for h in existing_handlers
    ]

    if main_log_path not in existing_paths:
        logger.add(
            Rank0Sink(open(main_log_path, "w")),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            filter=make_filter("main"),
        )

    for handler in existing_handlers:
        if getattr(handler._sink, "name", None) == sys.stderr:
            logger.remove(handler.id)
            logger.add(
                Rank0Sink(sys.stderr),
                format="{time:HH:mm:ss} | {level} | {message}",
                level="INFO",
                filter=make_filter("main"),
            )

    data_loggers = ["autocorr", "expectations", "populations"]
    if adaptive:
        data_loggers.append("bonddim")
    for name in data_loggers:
        add_file_logger(name, jobname)

    if size > 1:
        logger.add(
            sys.stderr,
            format=f"Rank{rank}:" + "{time:HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            filter=make_filter("rank"),
        )


def add_file_logger(name: str, jobname: str):
    """Add a new file logger

    Args:
        name: Name of the logger
        jobname: Job directory name

    Returns:
        logger: A logger that only writes to the specified data file
    """
    if not os.path.exists(f"./{jobname}"):
        os.makedirs(f"./{jobname}")

    log_path = f"{jobname}/{name}.dat"

    logger.add(
        Rank0Sink(open(log_path, "w")),
        format="{message}",
        level="DEBUG",
        filter=make_filter(name),
    )
