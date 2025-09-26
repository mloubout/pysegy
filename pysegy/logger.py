# Verbose printing control (defaults to disabled)
VERBOSE: bool = False


def set_verbose(verbose: bool = False) -> None:
    """Enable or disable verbose output for pysegy.

    Parameters
    ----------
    verbose : bool, optional
        When True, print informational messages; defaults to False.
    """
    global VERBOSE
    VERBOSE = bool(verbose)


def vprint(*args, **kwargs) -> None:
    """Print only when verbose mode is enabled."""
    if VERBOSE:  # pragma: no cover - trivial branch
        print(*args, **kwargs)
