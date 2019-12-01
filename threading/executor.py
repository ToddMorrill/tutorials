"""Illustrates how to use concurrent.futures to manage a pool of threads. Helps
write clean code since you don't have to worry about thread.join() statements.
Again, the scheduling of threads is done by the OS and does not follow a plan
that’s easy to figure out.
"""

import concurrent.futures
import logging
import time


def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function, range(3))