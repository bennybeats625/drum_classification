# ===============================
# ===============================
# Simple Stopwatch Implementation 
# ===============================
# ===============================

# =====================================
# (C) Copyright, 2020, Robert M. Nickel
# =====================================
# The contents of this file may not be shared without permission.

"""
The stopwatch module can be used via the following import:

>>> from stopwatch import tic, toc, time_string, wait

Measuring the completion time of a piece of code
can accomplished with:

>>> tic()
>>> # execute code here ...
>>> print('Processing time =',toc(),'sec')

A formatted time printout in terms of days, hours, minutes and
seconds is provided by:

>>> print('Processing time =',toc('s'))

or simply:

>>> toc('p')

Time measurements returned by toc() can be converted into a
formatted time string in terms of days, hours, minutes and
seconds with

>>> s = time_string(toc())

Suspension of program execution for t seconds is accomplished
with

>>> wait(t)

The stopwatch module is internally using the time module.
"""

# import the necessary modules
import time

# simple implementation of a wait command
def wait(sec=1):
    # simply wait the provided time in seconds
    time.sleep(sec)

# convert a number in seconds into a time string
def time_string(t):
    # initialize the output string
    s = ''
    # compute the days
    days = int(t/(60*60*24)); t = t - days*60*60*24
    if days > 0 : s = s + str(days) + ' days '
    # compute the hours
    hours = int(t/(60*60)); t = t - hours*60*60
    if hours > 0 : s = s + str(hours) + ' hrs '
    # compute the minutes
    mins = int(t/60); t = t - mins*60
    if mins > 0 : s = s + str(mins) + ' min '
    # compute the seconds
    secs = int(100*t)/100
    if secs > 0 : s = s + str(secs) + ' sec '
    # check for an empty string
    if len(s) == 0 : s = '0.0 sec '
    # return the time string
    return s

# start the stopwatch
def tic(s=None):
    # check the calling mode
    if s is None:
        # record the time
        tic.time = time.time()
        # and return
        return
    else:
        # return the recorded time
        return tic.time
    # end of tic

# stopwatch read out
def toc(s=None):
    # compute the time difference
    t = time.time()-tic('read')
    # check the calling mode
    if s is None:
        # return the time in seconds
        return t
    # convert the time to a formatted string
    t = time_string(t)
    # check if printout is desired
    if s[0] == 'p':
        # print the execution time
        print('Elapsed time: ' + t)
        return
    # return the time string
    return t

