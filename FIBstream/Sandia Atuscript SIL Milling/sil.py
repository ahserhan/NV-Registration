"""
Tutorial: Focused Ion Beam Spot Mill
"""
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client.structures import *

microscope = SdbMicroscopeClient()
microscope.connect('localhost')

# Try to adjust these two params in FIB GUI
# microscope.beams.ion_beam.horizontal_field_width.value=50e-6
# microscope.beams.ion_beam.beam_current.value=3e-9 #3nA

number_of_slice = 10
radius = 5e-6 # radius is 5 [micron]
x_offset = 0e-6 # shift pattern in x [micron]
y_offset = 0e-6 # shift pattern in y [micron]

for index in range(0, number_of_slice):
    microscope.patterning.clear_patterns()
    stream_file_path = 'sil_%02i.str'%index
    spd = StreamPatternDefinition.load(stream_file_path)

    print('loading sil_%02i...' % index, end='')
    pattern = microscope.patterning.create_stream(x_offset, y_offset, spd)
    print('done')

    defocus = radius/number_of_slice*index
    print(f'setting defocus to be {defocus}...', end='')
    pattern.defocus = defocus
    print('done')

    print('milling sil_%02i...'%index, end='')
    microscope.patterning.run()
    print('done\n')

microscope.patterning.clear_patterns()
print('All milling finished.')
