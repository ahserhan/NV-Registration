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

import numpy as np

class Path():
    
    """
    Defines a milling path. The constructor expects an Nx3 array, that
    describes a series of points (x,y,z).
    """

    def __init__(self, coordinates, depth_to_time):
        self.coordinates = coordinates
        self.depth_to_time = depth_to_time
    
    def longest_point(self):
        return int(self.coordinates[2].max()*self.depth_to_time)
    
    def plot(self,pos,tube_radius,rep=1,trunc=0,plot_rep=False):
        x,y,z = self.coordinates.copy()
        if trunc is not None:
            zp = z*(self.depth_to_time/rep)
            zp = zp.astype(np.int64)
            mask = zp >= trunc
            x = x[mask]
            y = y[mask]
            z = z[mask]
            n = len(x) # number of points (without additional blank points)
            i = len(mask)-n # number of points that were discarded during truncation
            if i > 0:
                print('Warning: %i of %i points discarded.'%(i,len(mask)))
        from mayavi import mlab
        if plot_rep:
            x += pos[0]
            y += pos[1]
            for n in range(rep):
                a=float(n+1)/rep
                mlab.plot3d(x,y,z*a+pos[2],z+pos[2],tube_radius=tube_radius, colormap='Spectral')
        else:
            x += pos[0]
            y += pos[1]
            z += pos[2]
            mlab.plot3d(x,y,z,z,tube_radius=tube_radius, colormap='Spectral')
    
    def render(self,pos,scale,rep=1,trunc=0):
        """
        Renders the path into a string and returns that string.
        The coordinates are shifted by adding 'pos',
        and subsequently scaled by multiplying by 'scale'.

        Asher's note: 
        Params @Scale:  is in pixel per micron.
        Params @depth_to_time:  is in 100 nanosecond per micron.
        Params @n:  number of points in this slice.
        """
        x,y,z = self.coordinates.copy()
        # print('\nHERE!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(np.min(x),np.max(x))
        # print(np.min(y),np.max(y))
        # print((pos[0],pos[1]))
        x += pos[0]
        y += pos[1]
        # print('\nHERE!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(np.min(x),np.max(x))
        # print(np.min(y),np.max(y))
        #FIXME: figure out the origin of the pixels and the direction.
        x = x * scale
        y = y * scale
        # print('\nHERE!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(np.min(x),np.max(x))
        # print(np.min(y),np.max(y))
        # print(np.min(z),np.max(z))
        z *= (self.depth_to_time/rep) # if we are using several rep, divide the milling times accordingly
        x = x.astype(np.int64)
        y = y.astype(np.int64)
        z = z.astype(np.int64)
        mask = z >= trunc # use the data only where it is larger or equal to the truncation value
        x = x[mask]
        y = y[mask]
        z = z[mask]
        # print('\nHERE!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        # print(np.min(x),np.max(x))
        # print(np.min(y),np.max(y))
        # print(np.min(z),np.max(z))
        t = np.sum(np.abs(z))*1e-7 # total milling time in seconds including blank points
        n = len(x) # number of points (without itional blank points)
        i = len(mask)-n # number of points that were discarded during truncation
        if i > 0:
            print('Warning: %i of %i points discarded.'%(i,len(mask)))
        c = np.vstack((z,x,y)).transpose()
        s = ''
        out_of_bounds = 0
        # FIXME: the condition also depends on where the origin of the pixel is
        for point in c:
            if not (0<=point[1]<65536 and 0<=point[2]<65535): # Asher:  it seems like only their x bound is 0 - 65536, but our bound is 65535 for both 
               out_of_bounds += 1
            s += '%i %i %i\n'%tuple(point)
        if out_of_bounds:
            print('Warning: %i points outside drawing area.'%out_of_bounds)        
            # s += '%i %i %i %i\n'%tuple(np.abs(z), x, y, np.sign(z))# ToDo: check whether this blanking works
        return s, n, t

class Scene():
    
    """
    Defines a Scene. A Scene is a rectangular drawing area
    with lower left and upper right corners (0,0), (width, width * 56574 / 2**16).
    The Scene holds the drawing elements.
    
    Parameters:
    
        width = the width of the Scene.
        
    *args:
    
        all additional unnamed parameters are interpreted as
        tuples (pos, item), representing drawing elements.
        They are added to the Scene.
    """

    def __init__(self, width, *args):
        self.items=[]
        self.width = width
        for arg in args:
            self.addItem(arg[0], arg[1])

    def addItem(self, item, pos):
        self.items.append( (item, np.array(pos)) )

    def plot(self, tube_radius, rep=1, trunc=True, plot_rep=False):
        if trunc is not None:
            rep,trunc = self.rep_and_trunc(rep)
        for item, pos in self.items:
            item.plot(pos,tube_radius=tube_radius,rep=rep,trunc=trunc,plot_rep=plot_rep)

    def longest(self):
        longest = 0
        for item, pos in self.items:
            longest=np.max( (longest,item.longest_point()) )
        return longest

    def rep_and_trunc(self, rep=1):
        """Determine the time range."""
        longest=self.longest()
        rep_min = longest/(2**18)+1 # (longest+1)/(4096*64)+1 
        if rep_min > rep:
            print( "Warning: the longest requested dwell time, "+str(longest/rep*1e-4)+"ms, exceeds the maximum dwell time, "+str(4096*64*1e-4)+"ms.")
            print( "Warning: increasing repetitions to "+str(rep_min)+" to reduce longest dwell time to "+str(longest/rep_min*1e-4)+"ms.")
            rep = rep_min
        longest=longest/rep
        # N=4096
        # n=0
        # while N<longest:
            # N=N<<1
            # n+=1
        # trunc = 2**n
        trunc=1
        print( "Info: using "+str(rep)+" repetitions.")
        print( "Info: longest dwell time is "+str(longest*0.1)+" micro s.")
        print( "Warning: truncating all milling points shorter than "+str(trunc*0.1)+" micro s.")
        return rep, trunc
        
    def render(self, rep=1):
        rep, trunc = self.rep_and_trunc(rep)
        scale = 2**16/self.width # Asher changed it from 12 to 16
        # print('scale = ',scale)
        N = 0
        T = 0
        s = ''
        for item, pos in self.items:
            sn, n, t = item.render(pos,scale,rep=rep,trunc=trunc)
            s += sn
            N += n
            T += t
        s = ('%i\n'%N) + s # Asher: third line would be total number of points
        s = ('%i\n'%rep) + s # Asher: second line would be repetition time
        s = 's16\n' + s # Asher: this is the first line of .str file, in our case this is 's16'
        # s = s[:-1] + ' 0\n' # Asher: I'm not sure why they do this (removing the last number and replace with ' 0')
        print( "Info: total number of points is %i."%N)
        print( "***************Info: total milling time is %.2f minutes.*************"%(T*rep/60.))
        if N > 8000000:
            print( "Error: total number of points exceeds 8 million.")
        return s
    
    def save(self, filename='scene.str', rep=1):
        s = self.render(rep=rep)
        f = open(filename,'w')
        f.write(s)
        f.close()