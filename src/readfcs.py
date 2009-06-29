import cStringIO
import operator
import struct
from warnings import warn
from fcmdata import FCMdata
from fcmannotation import Annotation
from utils import log2, fmt_integer, mask_integer

class FCSreader(object):
    """
    class to hold object to read and parse fcs files.  main usage is to get 
    a FCMdata object out of a fcs file
    """
    
    def __init__(self, filename, offset=0):
        self.filename = filename
        #self._fh = cStringIO.StringIO(open(filename, 'rb').read())
        self._fh = open(filename, 'rb')
        
        self._header = self.parse_header(offset)
        #parse text segment
        self.text = self.read_text(self._header['text_start'], self._header['text_end'])
        
        self.bitwidths = [int(getattr(self, 'P%dB' % i)) for i in range(1, int(self.size)+1)]
        self.ranges = [int(getattr(self, 'P%dR' % i)) for i in range(1, int(self.size)+1)]
        self.names = []
        self.altnames = []
        self.display = []
        self.max_value = []
        for i in range(1, int(self.size)+1):
            name = getattr(self, 'P%dS' % i)
            short_name = getattr(self, 'P%dN' % i)
            self.display.append(getattr(self, 'P%dDISPLAY' % i))
            self.max_value.append(getattr(self, 'P%dR' % i))
            if name and short_name:
                self.names.append(name + short_name)
            elif name:
                self.names.append(name)
            elif short_name:
                self.names.append(short_name)
            else:
                self.names.append('V%d' % (i+1))

            if name and short_name:
                self.altnames.append(name + ':' + short_name)
            if short_name:
                self.altnames.append(short_name)
            elif name:
                self.altnames.append(name)
            else:
                self.altnames.append('V%d' % (i+1))

        self.data = self.parse_data()
        
    def parse_header(self, offset):
        
        header = {}
        header['version'] = self.read_bytes(offset, 0, 5)
        header['text_start'] = int(self.read_bytes(offset, 10, 17))
        header['text_end'] = int(self.read_bytes(offset, 18, 25))
        header['data_start'] = int(self.read_bytes(offset, 26, 33))
        header['data_end'] = int(self.read_bytes(offset, 34, 41))
        header['analysis_start'] = int(self.read_bytes(offset, 42, 49))
        header['analysis_end'] = int(self.read_bytes(offset, 50, 57))
        
        return header
        
        
        
    def read_text(self, start, stop):
        text = self.read_bytes(start, stop)
        self.delim = text[0]
        if text[-1] != self.delim:
            warn('Text segment truncated!')
        text = [ i.strip('$') for i in text.strip(' ' + self.delim).split(self.delim)]
        self.text = dict(zip(text[::2], text[1::2]))
        self.size = int(self.text['PAR'])
        self.total = int(self.text['TOT'])
        self.mode = self.text['MODE']
        self.datatype = self.text['DATATYPE']
        self.order = self.text['BYTEORD']
    
    def read_bytes(self, offset, start, stop):
        """Read in bytes from start to stop inclusive."""
        self._fh.seek(offset+start)
        return self._fh.read(stop-start+1)
    
    def get_byte_order(self):
        if self.order == '1,2,3,4':
            byte_order = '<'
            #DEBUG -FCS FILES ON WINDOWS READ WRONGLY
            #byte_order = '>@'
        elif self.order == '4,3,2,1':
            byte_order = '>'
        else:
            # some strange byte order specified, use native byte order
            print "Unknown byte order: %s" % self.order
            byte_order = '@'
        return byte_order
    
    def parse_data(self):
        """Extract binary or other values from DATA section."""
        mode = self.mode
        first = self._header['data_first']
        last = self._header['data_last']
        if first == 0 and last == 0:
            first = int(self.text['BEGINDATA'])
            last = int(self.text['ENDDATA'])

        if mode == 'L':
            data = self.parse_list_data(first, last)
        elif mode == 'C':
            data = self.parse_correlated_data(first, last)
        elif mode == 'U':
            data = self.parse_uncorrelated_data(first, last)
        return data
    
    def parse_list_data(self, first, last):
        """Extract list data."""
        data_type = self.datatype
        byte_order = self.get_byte_order()
    
        if data_type == 'I':
            data = self.read_integer_data(byte_order, first, last)
        elif data_type in ('F', 'D'):
            data = self.read_float_data(byte_order, first, last)
        elif data_type == 'A':
            data = self.read_ascii_data(byte_order, first, last)
        else:
            print "Unknown data type: %s" % data_type
        return data

    def read_integer_data(self, byte_order, first, last):
        """Extract integer list data."""
        data = {}
        bitwidths, names, ranges = self.bitwidths, self.names, self.ranges
        unused_bitwidths = map(int, map(log2, ranges))
        # check that bitwidths can be handled by struct.unpack

        # DEBUG
        if reduce(operator.and_, [item in [8, 16, 32] for item in bitwidths]):
            print ">>> Regular"
            if len(set(bitwidths)) == 1: # uniform size for all parameters
                num_items = (last-first+1)/struct.calcsize(fmt_integer(bitwidths[0]))
                tmp = struct.unpack('%s%d%s' % (byte_order, num_items, fmt_integer(bitwidths[0])), 
                                                        self.read_bytes(first, last))
#                 if bitwidths[0]-unused_bitwidths[0] != 0: # not all bits used - need to mask
#                     bitmask = mask_integer(bitwidths[0], unused_bitwidths[0]) # assume same number of used bits
#                     tmp = [bitmask & item for item in tmp]
                for i, name in enumerate(names):
                    data[name] = tmp[i::len(names)]
                return data
            else: # parameter sizes are different e.g. 8, 8, 16,8, 32 ... do one at a time
                cur = first
                while cur < last:
                    for i, bitwidth in enumerate(bitwidths):
                        bitmask = mask_integer(bitwidth, unused_bitwidths[i])
                        nbytes = bitwidth/8
                        bin_string = self.read_bytes(cur, cur+nbytes-1)
                        cur += nbytes
                        val = bitmask & struct.unpack('%s%s' % (byte_order, fmt_integer(bitwidth)), bin_string)[0]
                        data.setdefault(names[i], []).append(val)
        else: # bitwidths are non-standard
            print "<<< Irregular"
            #TODO raise error
#            bits = bitbuffer.BitBuffer(open('../data/' + self.name + '.fcs', 'rb').read())
#            print ">>>", self.PAR, self.TOT
#            bits.seek(first*8)
#            for event in range(int(self.PAR)*int(self.TOT)):
#                i = event % int(self.PAR)
#                val = bits.readbits(bitwidths[i])
#                data.setdefault(names[i], []).append(val)
        return data
        
    def read_float_data(self, byte_order, first, last):
        """Faster version - reads in single chunk of float data."""
        data = {}
        names = self.names
        datatype = self.datatype.lower()
        num_items = (last-first+1)/struct.calcsize(datatype) 
        tmp = struct.unpack('%s%d%s' % (byte_order, num_items, datatype), self.read_bytes(first, last))
        for i, name in enumerate(names):
            data[name] = tmp[i::len(names)]
        return data
    
    def read_ascii_data(self, byte_order, first, last):
        """Extract ASCII list data."""
        data = {}
        names, bitwidths = self.names, self.bitwidths
        cur = first
        while cur < last:
            for i, bitwidth in enumerate(bitwidths):
                bin_string = self.read_bytes(cur, cur+bitwidth-1)
                cur += bitwidth
                data.setdefault(names[i], []).append(int(struct.unpack('%s%dB' % (byte_order, bitwidth), bin_string)[0]))
        return data
    
    def parse_correlated_data(self, first, last):
        """Extract correlated histogram data."""
        #TODO raise error
        return {}
    
    def parse_uncorrelated_data(self, first, last):
        """Extract uncorrelated histogram data."""
        #TODO: raise error.
        return {}
    
if __name__ == '__main__':
    foo = FCSreader('/home/jolly/Projects/flow/data/3FITC_4PE_004.fcs')
    #print foo.delim
    print foo.size
    print foo.total
    print foo.mode
    print foo.order
    print foo.text