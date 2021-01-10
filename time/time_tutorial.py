import time
import locale

# epoch time
print(time.gmtime(0))

# time since the epoch
t = time.time()
print(t)

# format the float representing the seconds since the epoch
print(time.ctime(t))

# create time from tuple
time_tuple = (2021, 1, 10, 8, 39, 55, 0, 10, 0)
t_struct = time.struct_time(time_tuple)
print(t_struct)

# access elements by name
print(t_struct.tm_year)

# local time
t_local = time.localtime()
print(t_local)
print(t_local.tm_zone)
print(t_local.tm_gmtoff)

# convert time to seconds
print(time.mktime(t_local))

# print local time as a string timestamp
print(time.asctime())

# format timestamp
print(time.strftime('%Y-%m-%d'))

# compare asctime to strftime
print(time.asctime())
locale.setlocale(locale.LC_TIME, 'zh_HK')
# will look the same as above
print(time.asctime())

# but using strftime, locale info will take effect
locale.setlocale(locale.LC_TIME, 'en_US')
print(time.strftime('%c'))
locale.setlocale(locale.LC_TIME, 'zh_HK')
print(time.strftime('%c'))
locale.setlocale(locale.LC_TIME, 'en_US')

# parse time from timestamp
print(time.strptime(time.ctime()))

# can use sleep to suspend execution of the program
print(time.ctime())
time.sleep(1)
print(time.ctime())

def longrunning_function():
    for i in range(1, 11):
        time.sleep(i / i ** 2)

start = time.perf_counter()
longrunning_function()
end = time.perf_counter()
print(end-start)