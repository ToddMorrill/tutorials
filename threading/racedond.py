"""Illustrates what a race condition is by simulating an update to a database.
TLDR; both threads are reading a value from a database, incrementing it locally,
and writing it back to the database. The issue is that the first thread cannot
write the incremented value before the second thread reads the value, resulting
in a only one incremement operation instead of 1. We need a locking mechanism.
"""

import concurrent.futures
import logging
import time


class FakeDatabase:
    def __init__(self):
        self.value = 0

    def update(self, name):
        logging.info("Thread %s: starting update", name)
        local_copy = self.value
        local_copy += 1
        time.sleep(0.1)
        self.value = local_copy
        logging.info("Thread %s: finishing update", name)


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    database = FakeDatabase()
    logging.info("Testing update. Starting value is %d.", database.value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            executor.submit(database.update, index)
    logging.info("Testing update. Ending value is %d.", database.value)