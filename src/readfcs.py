import cStringIO
from warnings import warn
from fcmdata import FCMdata
from fcmannotation import Annotation

class FCSreader(object):
    """
    class to hold object to read and parse fcs files.  main usage is to get 
    a FCMdata object out of a fcs file
    """
    
    def __init__(self, filename):
        self.filename = filename
        #self._fh = cStringIO.StringIO(open(filename, 'rb').read())
        self._fh = open(filename, 'rb')
        
        #parse headers
        self._header = {}
        self._header['version'] = self.read_bytes(0, 5)
        self._header['text_start'] = int(self.read_bytes(10, 17))
        self._header['text_end'] = int(self.read_bytes(18, 25))
        self._header['data_start'] = int(self.read_bytes(26, 33))
        self._header['data_end'] = int(self.read_bytes(34, 41))
        self._header['analysis_start'] = int(self.read_bytes(42, 49))
        self._header['analysis_end'] = int(self.read_bytes(50, 57))
        
        #parse text segment
        text = self.read_bytes(self._header['text_start'],self._header['text_end'])
        self.delim = text[0]
        if text[-1] != self.delim:
            warn('Text segment truncated!')
        text = [ i.strip('$') for i in text.strip(' ' + self.delim).split(self.delim)]
        self.text = dict(zip(text[::2], text[1::2]))
        self.size = int(self.text['PAR'])
        self.total = int(self.text['TOT'])
        self.mode = self.text['MODE']
        
        self.order = self.text['BYTEORD']
    
    def read_bytes(self, start, stop):
        """Read in bytes from start to stop inclusive."""
        self._fh.seek(start)
        return self._fh.read(stop-start+1)
    
if __name__ == '__main__':
    foo = FCSreader('/home/jolly/Projects/flow/data/3FITC_4PE_004.fcs')
    #print foo.delim
    print foo.size
    print foo.total
    print foo.mode
    print foo.order
    print foo.text