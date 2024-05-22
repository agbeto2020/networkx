import sys

__all__ = ["GraphMatcher", "DiGraphMatcher", "MultiGraphMatcher", "MultiDiGraphMatcher"]


class BitArray:
    """Implementation of bit array"""

    def __init__(self, _bytes_per_block, _size):
        _python_int_octect_step = sys.getsizeof(1) - sys.getsizeof(0)
        if _bytes_per_block % _python_int_octect_step != 0:
            print("")
            return
        self.bytes_per_block = _bytes_per_block
        self.bits_per_block = 8 * self.bytes_per_block
        # block value in decimale
        self.block_value = pow(2, self.bits_per_block) - 1
        self.bits = []
        self.nblocks = 0
        #
        if _size != None:
            curr_size = 0
            while curr_size < _size:
                self.bits.append(self.block_value)
                self.nblocks += 1
                curr_size += self.bits_per_block

    def set(self, ind, value):
        curr_size = self.nblocks * self.bits_per_block
        if ind >= 0:
            self.resize(ind + 1)
        if value != self.get(ind):
            block_index = ind // self.bits_per_block
            bit_index = ind % self.bits_per_block
            self.bits[block_index] ^= 1 << bit_index

    def get(self, ind):
        curr_size = self.nblocks * self.bits_per_block
        if ind >= 0 and ind < curr_size:
            block_index = ind // self.bits_per_block
            bit_index = ind % self.bits_per_block
            ## shift to the right
            return (self.bits[block_index] >> bit_index) & 1
        return 0

    def resize(self, new_size):
        curr_size = self.nblocks * self.bits_per_block
        while curr_size < new_size:
            self.bits.append(self.block_value)
            self.nblocks += 1
            curr_size += self.bits_per_block
