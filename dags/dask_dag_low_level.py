from dask.threaded import get
# from dask.multiprocessing import get
import subprocess


def task_one():
    cmd = ['python3', 'task_one.py']
    p = subprocess.Popen(cmd)
    status_code = p.wait()
    assert (status_code == 0)
    return status_code


def task_two():
    cmd = ['python3', 'task_two.py']
    p = subprocess.Popen(cmd)
    status_code = p.wait()
    assert (status_code == 0)
    return status_code


def task_three(*args):
    cmd = ['python3', 'task_three.py']
    p = subprocess.Popen(cmd)
    status_code = p.wait()
    assert (status_code == 0)
    return status_code


def main():

    d = {
        'task_one': (task_one, ),
        'task_two': (task_two, ),
        'task_three': (task_three, 'task_one', 'task_two')
    }

    print(get(d, 'task_three'))  # executes in parallel


if __name__ == '__main__':
    main()