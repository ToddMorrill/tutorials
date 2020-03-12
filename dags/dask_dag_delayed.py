import dask
import subprocess


@dask.delayed
def task_one():
    cmd = ['python3', 'task_one.py']
    p = subprocess.Popen(cmd)
    status_code = p.wait()
    assert (status_code == 0)
    return status_code


@dask.delayed
def task_two():
    cmd = ['python3', 'task_two.py']
    p = subprocess.Popen(cmd)
    status_code = p.wait()
    assert (status_code == 0)
    return status_code


@dask.delayed
def task_three(*args):
    cmd = ['python3', 'task_three.py']
    p = subprocess.Popen(cmd)
    status_code = p.wait()
    assert (status_code == 0)
    return status_code


def main():
    one = task_one()
    two = task_two()
    three = task_three(one, two)
    three.visualize(rankdir='LR')

    # executes in parallel, does not recompute intermediate steps
    # NB: since I'm already using subprocess, I can use scheduler='threads'
    # if I wasn't already using subprocesses and wanted physical parallelism
    # across multiple CPU cores, I would need scheduler='processes'
    print(dask.compute(one, two, three))


if __name__ == '__main__':
    main()