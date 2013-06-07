'''
Methods and object to generate alignments between datasets
'''

from fcm.alignment.align_data import DiagonalAlignData, CompAlignData, FullAlignData
from fcm.alignment.cluster_alignment import AlignMixture

__all__ = [ 'DiagonalAlignData',
           'CompAlignData',
           'FullAlignData',
           'AlignMixture',
           ]
