from __future__ import print_function
from RNABinding import *
from rgt.GenomicRegionSet import GenomicRegionSet

"""
Represent list of RNABinding.

Authors: Joseph Kuo

"""
class RNABindingSet(GenomicRegionSet):
    def __init__(self, name):
        """Initialize"""
        GenomicRegionSet.__init__(self, name = name)

    def sort(self):
        """Sort Elements by criteria defined by a GenomicRegion"""
        self.sequences.sort(cmp = GenomicRegion.__cmp__)
        self.sorted = True
    
if __name__ == '__main__':
    a = RNABindingSet(name="a")
    a.add(RNABinding("a",1,5))
    a.add(RNABinding("a",10,15))
    
    b = RNABindingSet(name="b")
    b.add(RNABinding("b",4,8))
    
    print(len(a))
    print(len(b))
    print(a.sequences)
    print(b.sequences)
    
    a.subtract(b)
    
    print(a.sequences)
    print(b.sequences)
    