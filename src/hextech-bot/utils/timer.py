"""Build the timefunc decorator."""

import time
from functools import wraps


def timefunc(verbose=True):
    """timefunc's doc"""

    def real_timefunc(function):
        @wraps(function)
        def time_closure(*args, **kwargs):
            """time_wrapper's doc string"""
            start = time.perf_counter()
            result = function(*args, **kwargs)
            time_elapsed = time.perf_counter() - start

            if verbose:
                print(f"Function: {function.__name__}, Time: {time_elapsed * 1000:.1f} ms")

            return result

        return time_closure

    return real_timefunc
