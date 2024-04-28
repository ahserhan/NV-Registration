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

def transform_target(o0, o1, o2, r, i0, i1, i2):

    """
    Compute the position of the NV in ion beam coordinates.
    
    Parameters:
    
        o0 =   marker 0 in optical coordinates
        o1 =   marker 1 in optical coordinates
        o2 =   marker 2 in optical coordinates
        r  =   target in optical coordinates

        i0 =   marker 0 in ion beam coordinates
        i1 =   marker 1 in ion beam coordinates
        i2 =   marker 2 in ion beam coordinates
        
    return:
    
        target in ion beam coordinates
    """

    v1 = o1-o0
    v2 = o2-o0
    r  = r-o0

    v1p = i1-i0
    v2p = i2-i0

    V = np.matrix((v1,v2)).transpose()
    Vp = np.matrix((v1p,v2p)).transpose()

    A = Vp*V.I

    i = np.dot(A,r)

    rp = i0+i

    return rp
### END transformations.py