### Notes on threading:
Following [this tutorial](https://realpython.com/intro-to-python-threading/)

1. Threads in Python do not actually execute concurrently as they would in C/C++. This is due to the limitations of the GIL.
1. Programs that are I/O bound or wait for remote events to occur may benefit from threading because certain threads can sleep while other threads can do something productive.
1. Programs that are CPU bound may not benefit from threading at all in Python due to the fact that threads do not run concurrently. In such a case, `multiprocessing` may be a better solution.
1. Threading can be helpful when considering how to design a program so that you can more easily reason about the control flow and layout of the program.