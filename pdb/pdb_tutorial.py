import pdb
import time

def add(a, b):
    val = a + b
    ## want to debug code here
    pdb.set_trace()
    return val

def main():
    a = 4
    b = 5
    print(add(a,b))

if __name__ == "__main__":
    main()