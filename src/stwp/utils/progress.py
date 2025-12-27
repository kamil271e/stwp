"""Progress bar utilities."""


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "\u2588",
    print_end: str = "\r",
) -> None:
    """Print a progress bar to the terminal.

    Call in a loop to create terminal progress bar.

    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Positive number of decimals in percent complete
        length: Character length of bar
        fill: Bar fill character
        print_end: End character (e.g. "\\r", "\\r\\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
    if iteration == total:
        print()
