"""
This file is part of FibStream.

    FibStream is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    FibStream is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FibStream.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2014 Helmut Fedder
"""

"""
This script generates stream file(s) for milling a SIL with a cone around it on an FEI Helios FIB.

Parameters:

    width        = width of the writing area [micron]

    mu           = milling rate [micron**3 / (micro s * nA )]
    I            = beam current [nA]
    ds           = step size in the x-y-plane [micron]

    R_sil        = radius of the SIL [micron]
    R_cone       = outer radius of the cone around the SIL [micron]
    N            = number of slices. The geometry is divided into N slices.
    rep          = number of repetitions. Note that the milling time on each pixel is
                   divided by the number of repetitons, i.e., the number of
                   repetitions just determines 'how smooth' the milling is done
                   and not how long.    

    position     = position of the structure on the writing area [micron]

Output:
    sil.str      = stream file

Usage example:

    First, decide what the size of the SIL and cone should be.
    The radius of the SIL is usually determined by the depth of the emitter around which we
    want to mill the SIL. The radius of the cone follows from the SIL radius and
    the N.A. of the objective.

    Second, decide how you want to mill the SIL. There are different strategies
    (see subsequent paragraph).

    Third, calibrate the milling speed. Go to some area and mill a series of test spirals,
    measure their depth and extract the milling speed in [micron / microsecond].
    
    Note: a) when milling the test structure, use the same 'ds' that you want to
             use later for milling the SIL (also same beam current, magnification,
             focus, etc.), otherwise you have to scale the 'mu' accordingly!
             
          b) the milling speed should depend on the number of repetitions, thus,
             when milling the test structure, try different values for the number
             repetitions and see how that effects the effective milling speed!

    Fourth, calibrate the image size. Assuming that the machine is well calibrated, the
    absolute image size (expressed also by the scale bar) is known. The precise number
    for the total image size is available from the control software. Find that
    number (width and width of the image) and enter it (the width [micron]) in the
    script. Now, all your SILs will come out with the right absolute size.


How to mill a SIL?

    The SIL with the cone is milled by moving the beam along a 'double spiral' path
    (see figure) and adjusting the dwell time at each point to create the desired
    radial profile.

    The beam starts always on the outside, spirals inwards and then spirals outwards
    again.
    
    In order to ablate material smoothly, it is important to write the path not
    only once, but rather write it many times. There are mainly three distinct ways
    to do this.

        1. use the same path multiple times, i.e., simply introduce a 'number of repetitions'
           and divide the milling time for a single pass by this number
           
           Pros: easy
           Cons: the ratio between the largest and the shortest milling time must not exceed
                 4096, thus, short milling times - corresponding to the tip of the sil - 
                 are rounded and the SIL gets steps and un-milled area at the tip
        
        2. Use several different paths, i.e., divide the SIL into several different
           slices and use the suitable milling path for each slice.
           
           Pros: steps in z-direction can be made arbitrarily fine
                 can still be combined with multiple repetitions, enabling very fine milling
           Cons: are there any?

        3. same as 2. but in addition use optimal beam focus in each layer.

           Pros: enables milling with high lateral resolution in each layer, i.e.,
                 prevents blurring of the structure at large depths.
           Cons: Requires multiple stream files, relies on autoFIB or NanoBuilder features.
                 Due to the stepwise change of the focus for each layer, this may still
                 lead to steps in z-direction
           
    The present script can generate stream files for each of the three approaches. In the
    1. and 2. approach, a single stream file is sufficient (or multiple files is the number
    of points exceeds the maximum allowed value), the last approach requires
    several stream files that need to be fed into the autoFIB or NanoBuilder. Additionally
    the machine needs the information how much it should change the focus / move in z-direction
    for each layer.
"""

import numpy as np

from patterns import sil_with_cone
from stream import Path, Scene
from rates import mu_diamond as mu # milling rate [micrometer**3 / (microsecond * nA)] 

############################################
# SIL shape
############################################

############################################
# relation between N.A. and cone radius
############################################
#  N.A. |  R_cone/R_sil         # R_cone/R_sil = np.tan(np.arcsin(NA))
#       |
#  0.8  |   1.33
#  0.85 |   1.62
#  0.9  |   2.1
#  0.95 |   3.1  

# radius of the SIL [micron]
R_sil = 5.0
R_cone = 2.2*R_sil  # outer radius of the cone [micron]

print("Radius of the SIL: "+str(R_sil))

############################################
# ion beam parameters
############################################

#I = 0.92 # current [nA]
I = 2.8 # current [nA]
# I = 0.92 # current [nA]
ds = 0.05 # step size in the x-y-plane [micron]

# conversion factor to convert z into milling time [micro s]
zeta = mu * I / ds**2

print("milling rate in micron / micro s is :"+str(zeta))
############################################
# width of the image and position of structure
############################################

#width = 36.5 # x 3.500 width of the drawing area [micron]
#width = 10.65 # x 12.000 width of the drawing area [micron]
#width = 25.6 # x 5.000 width of the drawing area [micron]
width = 64.0
position = (0.,0.) # center of the SIL

dz=0.05 # thickness of each slice [micron]
N=int(R_sil/dz) # number of slices

print("Info: using "+str(N)+" slices of thickness "+str(dz)+" micron.")

slices = sil_with_cone(R_sil, R_cone, ds, N=N) # compute the milling paths

# split into several stream files if necessary
k_max=7000000
j=0
k=0
scene = Scene( width=width )
for n,sli in enumerate(slices):
    dk=sli.shape[1]
    if k+dk > k_max:
        fname = 'sil_%02i.str'%j
        scene.save(fname, rep=1)
        scene = Scene( width=width )
        j += 1
        k = 0
    scene.addItem( Path(sli,10./(zeta*2.)), np.append(position,n*dz) )
    k += dk
if k > 0:
    fname = 'sil_%02i.str'%j
    scene.save(fname, rep=1)

"""
# plot 3D path
from mayavi import mlab

for n in range(N):        
    x,y,z = slices[n]
    zp = z+n*dz
    mlab.plot3d(x,y,zp,zp,tube_radius=0.1*ds, colormap='Spectral')
"""

"""
#########################################
# Approach 3
#
# Write a stream file for each layer and a script file that runs all files
#########################################
directory = ''
f = open(directory+'sil.psc','w')
for n, slice in enumerate(slices):
    # write slice to stream file
    stream_file_name = 'sil_%02i.str'%n
    scene = Scene( width=width )
    slice[2] *= 10./(zeta*2.) # convert the z-coordinate [micron] into milling time in maschine units [100nm].
    scene.addItem( Path(slice), position )
    scene.save(directory+stream_file_name, rep=rep)

    # add line for this layer to milling script
    f.write('clear\n')
    f.write('streamfile "'+stream_file_name+'"\n')
    f.write('mill\n')
    if adjust == 'stage':
        f.write('stagemovedelta z, %f\n'%dz*1e-3)
    elif adjust == 'focus':
        f.write('getfocus\n')
        f.write('newfocus = focus + %f\n'%dz)
        f.write('setfocus newfocus\n')

f.close()
"""