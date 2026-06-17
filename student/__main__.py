import sys
from .cli import main as cli_main


_default_unraisablehook = sys.unraisablehook


def _suppress_event_loop_del_noise(
      unraisable: "sys.UnraisableHookArgs") -> None:
    """Silence the harmless asyncio loop teardown traceback at exit.

    `litellm`/`dspy` spawn a background event loop that is only
    garbage-collected at interpreter shutdown, by which point its
    self-pipe file descriptors may already be closed. This raises a
    `ValueError("Invalid file descriptor: -1")` inside
    `BaseEventLoop.__del__`, which Python reports via
    `sys.unraisablehook` instead of a normal exception. It's cosmetic
    and doesn't affect program correctness, so we filter only this
    exact case and let any other unraisable error print as usual.
    """

    if (
        unraisable.exc_type is ValueError
        and "Invalid file descriptor" in str(unraisable.exc_value)
    ):
        return
    _default_unraisablehook(unraisable)


sys.unraisablehook = _suppress_event_loop_del_noise


def main() -> None:
    cli_main()

if __name__ == "__main__":
    main()
