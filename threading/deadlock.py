"""Illustrates what happens when deadlock occurs, usually when a lock hasn't
been released and there is a circular dependency on two resources.
"""

import threading

l = threading.Lock()
print("before first acquire")
l.acquire()
print("before second acquire")
l.acquire()
print("acquired lock twice")