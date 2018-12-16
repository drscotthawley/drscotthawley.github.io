---
layout: post
title: Parallelizing Python, Simplified
subtitle: Tips for a few common tasks
description: Basic use of multiprocessing 
excerpt: So you have some serial task that takes forever, and you're thinking it should be parallelizable, but you find
the documentation on this to be obtuse?  Yea. 
image: ../images/parallelpython.png
bg-image: ../parallelpython.jpg
comments: true
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 

# Parallelizing Python, Simplified

So you have some serial task that takes forever, and you're thinking it should be parallelizable, but you find
the documentation on this to be obtuse?  Yea. 

Usually I'm interested in either *outputting* lots of data in parallel, or *inputting* lots of data in parallel, and it's
usually something that I first implemented as a loop but got tired of how slow it runs.

There's a simple prescription for parallelizing most of these kinds of tasks.  It goes as follows:

0. Have some kind of task performed in a for loop. 
1. Write a function that does what you want for one "instance."  For example, take what's inside one of your for loops,
put all that in a separate function.
2. Keep your loop but use only the function call. Make sure it produces the same results as the original version of your code.
3. Use `functools.partial` to create a wrapper for your function.
4. Replace the loop with a call to `Pool.map()`. 

Let's do a few examples:

## Example 1: Generate a bunch of files
Let's say you have some important synthetic data that you want to generate lots of instances of. 
For now, for simplicity, we're just going to generate images of, let's say, random noise.  And to make it interesting 
we'll generate 2000 of them. 

Here's the serial for-loop version:
~~~ python
import numpy as np
import cv2

n_images = 2000
size_x, size_y = 100, 100
for i in range(n_images):
    arr = 255*np.random.rand(size_x,size_y)
    filename = 'image_'+str(i)+'.png'
    print("writing file ",filename)
    cv2.imwrite(filename,arr)
~~~

Now we write dedicated function, put it in a `partial` wrapper, and call it as follows:
~~~ python
import numpy as np
import cv2
from functools import partial

def write_one_file(size_x, size_y, name_prefix, index):
    arr = 255*np.random.rand(size_x,size_y)
    filename = name_prefix + str(index) + '.png'
    print("writing file ",filename)
    cv2.imwrite(filename,arr)

n_images = 2000
size_x, size_y = 100, 100

wrapper = partial(write_one_file, size_x, size_y, 'image_')
for i in range(n_images):
    wrapper(i)
~~~

Finally we replace the loop with a multiprocessing pool. We can either use all the cpus on the machine (which is the default) 
or specify how many to use, by giving an argument to `Pool()`:

~~~ python
import numpy as np
import cv2
from functools import partial
import multiprocessing as mp

def write_one_file(size_x, size_y, name_prefix, index):
    arr = 255*np.random.rand(size_x,size_y)
    filename = name_prefix + str(index) + '.png'
    print("writing file ",filename)
    cv2.imwrite(filename,arr)

n_images = 2000
size_x, size_y = 100, 100

wrapper = partial(write_one_file, size_x, size_y, 'image_')

num_procs = mp.cpu_count() # or can replace with some number of processes to use
pool = mp.Pool(num_procs)
indices = range(n_images)
results = pool.map(wrapper, indices)
pool.close()
pool.join()
~~~

There are other ways you can do this to get more control, e.g. to have each process in the pool receive a particular 
*range* of indices, but this basic setup will get the job done.  And if you turn off the printing to screen and time the execution,
you'll see the speedup.



## Example 2: Read a bunch of files into a list
This example is actually of limited utility and you may want to just skip down to "Example 3: Filling a numpy array," but 
it's still an illustrative example that motivates Example 3, and offers a bit of variety in how one might do things.  In this case we're *not* going to use Pool.map; instead we're going to use a context manager for the particular datatype of `list`.  

Let's try to load in all the image files we just generated, into a list.  We'll split up the tasks manually into equal numbers for each process.
~~~ python
import glob
import cv2

name_prefix = 'image_'
# we'll use glob to get the list of available files
# note that glob order isn't...easily discernible, so we'll sort.
img_file_list = sorted(glob.glob(name_prefix+'*.png'))
n_files = len(img_file_list)
print(n_files,"files available.")

img_data_list = []
for i in range(n_files):
    filename = name_prefix + str(i) + '.png'
    print("Reading file",filename)
    img = cv2.imread(filename)
    img_data_list.append(img)

print(len(img_data_list),"images in list.")
~~~
If we wanted to, we could easily convert this list of images to a numpy array. But let's hold off on that.  Parallelizing this looks like the following process:

~~~ python
from multiprocessing import Process, Manager, cpu_count
import glob
import cv2


def load_one_proc(img_data_list, img_file_list, iproc, per_proc):
    istart, iend = iproc * per_proc, (iproc+1) * per_proc
    for i in range(istart,iend):    # each process will read a range of files
        filename = img_file_list[i]
        print("Reading file",filename)
        img = cv2.imread(filename)
        img_data_list.append(img)
    return

name_prefix = 'image_'
# we'll use glob to get the list of available files
# note that glob order isn't...easily discernible, so we'll sort.
img_file_list = sorted(glob.glob(name_prefix+'*.png'))
n_files = len(img_file_list)
print(n_files,"files available.")

# We'll split up the list manually
num_procs = cpu_count()
print("Parallelizing across",num_procs,"processes.")

per_proc = n_files // num_procs  # Number of files per processor to load
assert n_files == per_proc * num_procs  # Obvously one can do something more sophisticated than this!

with Manager() as manager:
    img_data_list = manager.list()
    processes = []
    for iproc in range(num_procs):
        p = Process(target=load_one_proc, args=(img_data_list, img_file_list, iproc, per_proc))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    outside_list = list(img_data_list)   # Copy out of the Manager context (there may be a better way to do this)

print(len(outside_list),"images in list.")
~~~
