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
To mill a SIL, we use an Archimedes spiral that starts at (0,0) and
spirals outwards. The important property of an Archimedes spiral that
is relevant here is that the distance between neighbouring lines is
constant (for details about the Archimedes spiral see also mathworld.com).

For an Archimedes spiral, the radial coordinate r and
the polar angle t ('theta') are related by

    r=a*t,

where 2*pi*a is the 'spurweite' (distance betwen neighbouring lines).

The arclength s is

    s = 1/2*a*(t*(1+t**2)**0.5 + ln(t+(1+t**2)**0.5))

TO obtain a relation between the step width ds on the arc and the
angular step width dt, we evaluate the differential

    ds = a*(1+t**2)**0.5 * dt

The milling depth z follows from the radius R and the planar
radial coordinate r

    z = R-(R**2-r**2)**0.5

We introduce the ratio alpha between the 'spurweite' and
the step width on the arc, such that

    a = alpha*ds/(2*pi)

We cut the SIL into horizontal slices of thickness h. The first slice is
at the top of the SIL. The horizontal cuts define N circles in the
x-y-plane with radii

    rn = (R**2-(R-n*h)**2)**0.5
    
The milling process is as follows. In slice 0, we mill the entire circle, however
we use reduced milling time in the center, resembling the SIL profile and
constant milling time outside the circle r0. In slice 1, we start milling at radius
r0. In the ring between r0 and r1, we mill the SIL profile, outside r1, we use constant milling
time to remove the layer. In this way, we set up the milling process for the entire SIL.

We aim to express everything in terms of t. We introduce

    T = R/a
    tn = rn/a = (R**2-(R-n*h)**2)**0.5 / a = (T**2-(T-n*h/a)**2)**0.5 = (2*n*h/a - (n*h/a)**2)**0.5

The cartesian coordinates are

    x = r*cos(t) = a*t*cos(t)
    y = r*sin(t) = a*t*sin(t)
    z = R-(R**2-r**2)**0.5 = a*(T-(T**2-t**2)**0.5)
    dt = ds / (a*(1+t**2)**0.5) = 2*pi / (alpha*(1+t**2)**0.5)  

Since x,y,z are proportional to 'a', and 'a' is not included in the expression for dt,
we can evaluate all expressions in dimensionless units and just multiply the final
resulting arrays by 'a'. 

To allocate arrays, we need the number of points N in the spiral

    N = S/ds = 0.5*alpha/(2.*np.pi)*( T*(1+T**2)**0.5 + np.log(T+(1+T**2)**0.5) ))
    
The strategy for generating the slices is now as follows.

    1. Using the angular coordinate t, we generate an Archimedes spiral and milling times
       for the entire SIL. While doing this, we evaluate the array indices i_n where the
       planar radius r crosses the slice radii rn.
    2. To construct slice n, we take the portion of the spiral coordinates x,y that
       is larger than rn, i.e., starting from index i_n and to the end, and we take the
       milling times in a ring between i_n and i_n+1 and constant milling time for
       larger radii and we subtract the milling time for n-1 layers.
"""

import numpy as np

def cross(l,w,d):

    """
    Path for a cross.
    
    Parameters:
        
        l = length of cross
        w = width of line (beam diameter)
        d = distance between two milling points
    """

    r = np.arange(-0.5*l,0.5*l,d)
    x = np.append(r,np.zeros_like(r))
    y = np.append(np.zeros_like(r),r)
    z = np.ones_like(x)
    
    return np.vstack( (x,y,z) )

def grid(T,N):
    """
    Path for a square grid.
    
    Parameters:
    
        T = distance between two points
        N = number of points in both x- and y-direction
    """
    g = np.meshgrid(T*np.arange(N),T*np.arange(N))

    x = g[0].flatten()
    y = g[1].flatten()
    z = np.ones_like(x)

    return np.vstack( (x,y,z) )
    

def spiral(R, ds, alpha=1.0):
    """
    Path for a growing spiral.

    Parameters:
    
        R    = radius of the spiral [micron]
        ds   = step width in the x-y-plane [micron]
    """


    a = alpha*ds/(2.*np.pi)
    T = R/a
    K = int(0.5*alpha/(2.*np.pi)*( Tc*(1+Tc**2)**0.5 + np.log(Tc+(1+Tc**2)**0.5) ))

    x = np.empty(K)
    y = np.empty(K)

    t = 0
    k = 0
    while t<T:
        x[k] = t*np.cos(t)
        y[k] = t*np.sin(t)
        
        t+=2*np.pi/(alpha*(1.0+t**2)**0.5)
        k+=1

    x = a*x[:k]
    y = a*y[:k]
    z = np.ones_like(x)

    return np.vstack( (x,y,z) )

def double_spiral(R, ds, alpha=1.0):
    """
    Path for a double spiral.

    Parameters:
    
        R    = radius of the spiral [micron]
        ds   = step width in the x-y-plane [micron]
    """
    
    a = alpha*ds/(2.*np.pi)
    T = R/a
    K = int(0.5*alpha/(2.*np.pi)*( T*(1+T**2)**0.5 + np.log(T+(1+T**2)**0.5) ))

    x = np.empty(K)
    y = np.empty(K)

    t = 0
    k = 0
    while t<T:
        x[k] = t*np.cos(t)
        y[k] = t*np.sin(t)
        
        t+=2*np.pi/(alpha*(1.0+t**2)**0.5)
        k+=1

    x = a*x[:k]
    y = a*y[:k]
    z = np.ones_like(x)

    x = np.append(x[::-1],-x[1:])
    y = np.append(y[::-1],-y[1:])
    z = np.append(z[::-1],z[1:])

    return np.vstack( (x,y,z) )

def spiral_ring(R_inner, R_outer, ds, alpha=1.0):
    """
    Path for a double spiral.

    Parameters:
    
        R_inner = inner radius of the spiral ring [micron]
        R_outer = outer radius of the spiral ring [micron]
        ds   = step width in the x-y-plane [micron]
    """
    
    a = alpha*ds/(2.*np.pi)
    T_inner = R_inner/a
    T_outer = R_outer/a
    K = int(0.5*alpha/(2.*np.pi)*( T_outer*(1+T_outer**2)**0.5 + np.log(T_outer+(1+T_outer**2)**0.5) ))

    x = np.empty(K)
    y = np.empty(K)

    t = 0
    k = 0
    while t<T_inner:
        t+=2*np.pi/(alpha*(1.0+t**2)**0.5)
    while t<T_outer:
        x[k] = t*np.cos(t)
        y[k] = t*np.sin(t)
        
        t+=2*np.pi/(alpha*(1.0+t**2)**0.5)
        k+=1

    x = a*x[:k]
    y = a*y[:k]
    z = np.ones_like(x)

    return np.vstack( (x,y,z) )

def sil(R, d, N, mu=1.0, alpha=1.0):
    """    
    R = SIL radius in base units
    h = thickness of one layer in base units (if R/h is not an integer, actual thickness is adjusted to the nearest integer)
    d = planar step size in base units
    
    mu = milling time in microseconds per base unit
    
    alpha = ratio between spurweite and d (alpha=1.0 --> spurweite == d)
    """
    
    a = alpha*d/(2.*np.pi)
    T = R/a
    K = int(0.5*alpha/(2.*np.pi)*( T*(1+T**2)**0.5 + np.log(T+(1+T**2)**0.5) ))
    
    dz = R/N # thickness of one slice 
    tz = dz/a # thickness of one slice in units of theta
    
    x = np.empty(K)
    y = np.empty(K)
    z = np.empty(K)
    i = [ 0 ]

    t = 0
    k = 0
    for n in range(1,N+1):
        tn = (2*n*tz*T - (n*tz)**2)**0.5
        while t<tn:
            x[k] = t*np.cos(t)
            y[k] = t*np.sin(t)
            z[k] = T-(T**2-t**2)**0.5
             
            t+=2*np.pi/(alpha*(1.0+t**2)**0.5)
            k += 1
        i += [ k ]

    x = a*x[:k]
    y = a*y[:k]
    z = a*z[:k]
    
    slices = []
    
    for n in range(len(i)-1):
        xn = x[i[n]:]
        yn = y[i[n]:]
        zn = np.append(z[i[n]:i[n+1]]-n*dz, dz*np.ones(i[-1]-i[n+1])) 
        slices += [ np.vstack( (xn,yn,zn) ) ]
    
    return slices

def cone(R1, R2, H, d, alpha=1.0):

    a = alpha*d/(2.*np.pi)
    T1 = R1/a
    T2 = R2/a
    TZ =  H/a
        
    x = []
    y = []
    z = []

    t = T1
    while t<T2:
        x += [ t*np.cos(t) ]
        y += [ t*np.sin(t) ]
        z += [ (T2-t)/(T2-T1)*TZ ]
         
        t+=2*np.pi/(alpha*(1.0+t**2)**0.5)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    #return np.vstack( (x,y,m) )
    return a * np.vstack( (x[::-1],y[::-1],z[::-1]) )


def sil_with_cone(R_sil, R_cone, ds, N, alpha=1.0):
    """
    Path to mill a SIL together with a cone at once.
    
    The path starts and ends at (0,0) and consists of a spiral that first goes
    outwards and then inwards.
    """

    a = alpha*ds/(2.*np.pi)
    Ts = R_sil/a
    Tc = R_cone/a
    K = int(0.5*alpha/(2.*np.pi)*( Tc*(1+Tc**2)**0.5 + np.log(Tc+(1+Tc**2)**0.5) ))

    dz = R_sil/N # thickness of one slice 
    tz = dz/a # thickness of one slice in units of theta
    
    x = np.empty(K)
    y = np.empty(K)
    theta = np.empty(K)
    
    # generate spiral, keeping the angular mesh
    k = 0
    t = 0
    while t<Tc:
        x[k] = t*np.cos(t)
        y[k] = t*np.sin(t)
        theta[k] = t
        
        t+=2*np.pi/(alpha*(1.0+t**2)**0.5)
        k+=1
        
    x = x[:k]
    y = y[:k]
    theta = theta[:k]

    # create the planar radii for the slices (radii are measured in units of theta)
    ns = np.arange(N+1)
    t_sil = (2*ns*tz*Ts - (ns*tz)**2)**0.5
    t_cone = Ts+(Tc-Ts)/N*ns[::-1]

    # indices in the theta mesh, where the planar radii intersect with the spiral
    i_sil = theta.searchsorted(t_sil)
    i_cone = theta.searchsorted(t_cone)

    # milling depth
    z_sil = Ts-(Ts**2-theta[:i_sil[-1]]**2)**0.5
    z_cone = (Tc-theta[i_sil[-1]:])/(Tc-Ts)*Ts
    
    z = np.append(z_sil,z_cone)
    
    x = a*x
    y = a*y
    z = a*z
    
    slices = []
    
    for n in range(N):
        xn = x[i_sil[n]:i_cone[n]]
        yn = y[i_sil[n]:i_cone[n]]
        zn = np.append(z[i_sil[n]:i_sil[n+1]]-n*dz,dz*np.ones(i_cone[n+1]-i_sil[n+1]))
        zn = np.append(zn,z[i_cone[n+1]:i_cone[n]]-n*dz)

        if n == 0:
            xn = np.append(xn[::-1],-xn[1:])
            yn = np.append(yn[::-1],-yn[1:])
            zn = np.append(zn[::-1],zn[1:])
        else:
            xn = np.append(xn[::-1],-xn)
            yn = np.append(yn[::-1],-yn)
            zn = np.append(zn[::-1],zn)
        slices += [ np.vstack((xn,yn,zn)) ]
    
    return slices    


if __name__ == '__main__':
    
    """
    This script generates stream files for milling a SIL woth a cone around it.
    
    Parameters:
    
        size         = size of the writing area [micron]
        R_sil=10.    = radius of the SIL [micron]
        R_cone=20.   = outer radius of the cone around the SIL [micron]
        N=1          = number of slices for milling the SIL
        
        ds=0.10      = step width in the x-y-plane [micron]
        mu=10.0      = milling speed [micron / microsecond]

    Output:
        sil_01.txt   = first (top) layer of the structure.
            .
        sil_<N>.txt  = last (bottom) layer of the structure.
    """
    
    size = 64 # size of the drawing area [micron]
    R_sil = 5 # radius of SIL [micron]
    R_cone = 11. # outer radius of the cone [micron]
    ds = 0.05 # step width in the x-y-plane [micron] 
    N = 1 # number of slices
    mu = 1.0 # milling speed [micron / microsecond]
    depth_to_time = 1.1e5# [100* nanosecond / micron] Asher:This gives the first entry with unit 100* ns
    
    # generate the milling path in real space coordinates
    slices = sil_with_cone(R_sil, R_cone, ds, N)
    
    from stream import Path, Scene    
    
    # convert path to streamfile
    for n, slice in enumerate(slices):
        # slice[2] *= mu  # convert the milling depth [micron] into milling time. Asher: seems like we don't need it here 
        scene = Scene( width=size ) # Asher Revised size -> width
        scene.addItem( Path(slice, depth_to_time),(0.5*size,0.5*size)) # place the structure in the center of the drawing area
        # Asher added the mu argument in the Path function above, removed the transpose method, and switched the two arguments above
        scene.save('sil_%02i.str'%n)
    
    # plot 3D path

    # from mayavi import mlab
    
    # dz = R_sil/N # thickness of one slice
    # for n in range(N):        
    #     x,y,z = slices[n]
    #     zp = z+n*dz
    #     mlab.plot3d(x,y,zp,zp,tube_radius=0.1*ds, colormap='Spectral')
    #     mlab.show()