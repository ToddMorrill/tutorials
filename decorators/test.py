import math
from decorators import do_twice, timer, debug, slow_down, cache, count_calls

@do_twice
def say_whee():
    print("Whee!")

@do_twice
def greet(name):
    print(f"Hello {name}")

@do_twice
def return_greeting(name):
    print("Creating greeting")
    return f"Hi {name}"

@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])

# Apply a decorator to a standard library function
math.factorial = debug(math.factorial)

def approximate_e(terms=18):
    return sum(1 / math.factorial(n) for n in range(terms))

@slow_down
def countdown(from_number):
    if from_number < 1:
        print("Liftoff!")
    else:
        print(from_number)
        countdown(from_number - 1)

@slow_down(rate=2)
def countdown(from_number):
    if from_number < 1:
        print("Liftoff!")
    else:
        print(from_number)
        countdown(from_number - 1)

class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """Get value of radius"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Set radius, raise error if negative"""
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius must be positive")

    @property
    def area(self):
        """Calculate area inside circle"""
        return self.pi() * self.radius**2

    def cylinder_volume(self, height):
        """Calculate volume of cylinder with circle as base"""
        return self.area * height

    @classmethod
    def unit_circle(cls):
        """Factory method creating a circle with radius 1"""
        return cls(1)

    @staticmethod
    def pi():
        """Value of π, could use math.pi instead though"""
        return 3.1415926535

class TimeWaster:
    @debug
    def __init__(self, max_num):
        self.max_num = max_num

    @timer
    def waste_time(self, num_times):
        for _ in range(num_times):
            sum([i**2 for i in range(self.max_num)])

# @count_calls

@cache
@debug
def fibonacci(num):
    if num < 2:
        return num
    return fibonacci(num - 1) + fibonacci(num - 2)

if __name__ == '__main__':
    # say_whee()

    # greet('world')
    
    # return_val = return_greeting('Adam')
    # print(return_val)
    
    # waste_some_time(1)
    # waste_some_time(999)
    
    # approximate_e(5)
    
    # countdown(10)
    
    # c = Circle(5)
    # print(c.radius)
    # print(c.area)
    # c.radius = 2
    # print(c.area)
    # try:
    #     c.area = 100
    # except:
    #     pass
    # c.cylinder_volume(height=4)
    # try:
    #     c.radius = -1
    # except:
    #     pass
    # c = Circle.unit_circle()
    # print(c.radius)
    # print(c.pi())
    # print(Circle.pi())
    
    # tw = TimeWaster(1000)
    # tw.waste_time(999)

    # countdown(3)

    print(fibonacci(10))