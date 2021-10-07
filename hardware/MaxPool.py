import math
from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable

@dataclass
class MaxPool:
    rows: int
    cols: int
    channels: int
    kernel_size: int
    stride: int
    pad: int
    streams_in: GKVariable = None
    streams_out: GKVariable = None

    @property
    def rows_out(self):
        return int(math.floor((self.rows-self.kernel_size+2*self.pad)/self.stride)+1)

    @property
    def rows_out(self):
        return int(math.floor((self.cols-self.kernel_size+2*self.pad)/self.stride)+1)

    def latency_in(self):
        return self.rows*self.cols*self.channels/self.streams_in

    def latency_out(self):
        return self.rows_out*self.cols_out*self.channels/self.streams_out

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }

