
from warnings import warn
from fcmdata import FCMdata
from fcmannotation import Annotation
import re


class FCSreader(object):
    """
    class to hold object to read and parse fcs files.  main usage is to get 
    a FCMdata object out of a fcs file
    """
    def __init__(self, filename):
        self.filename = filename
        #self._fh = cStringIO.StringIO(open(filename, 'rb').read())
        self._fh = open(filename, 'rb')
        self.cur_offset = 0
        
        
        
   
    def get_FCMdata(self):
        # parse headers
        header = self.parse_header(self.cur_offset)
        # parse text 
        text = self.parse_text(self.cur_offset, header['text_start'], header['text_stop'])
        # parse annalysis
        try:
            astart = text['beginanalysis']
        except KeyError:
            astart = header['analysis_start']
        try:
            astop = text['endanalysis']
        except KeyError:
            astop = header['analysis_end']
        analysis = self.parse_analysis(self.cur_offset, astart, astop)
        # parse data
        try:
            dstart = text['beginadata']
        except KeyError:
            dstart = header['data_start']
        try:
            dstop = text['enddata']
        except KeyError:
            dstop = header['data_end']
        
        data = self.parse_data(self.cur_offset, dstart, dstop, text)
        # build fcmdata object
        channels = []
        scatters = []
        tmpfcm = FCMdata(data, channels, scatters,
            Annotation({'text': text, 'header': header, 'analysis': analysis}))
        # update cur_offset for multidata files
        # return fcm object
        return tmpfcm
        
    
    def read_bytes(self, offset, start, stop):
        """Read in bytes from start to stop inclusive."""
        self._fh.seek(offset+start)
        return self._fh.read(stop-start+1)
    
    def parse_header(self, offset):
        """
        Parse the FCM data in fcs file at the offset (supporting multiple 
        data segments in a file
        """
        header = {}
        header['text_start'] = int(self.read_bytes(offset, 10, 17))
        header['text_stop'] = int(self.read_bytes(offset, 18, 25))
        header['data_start'] = int(self.read_bytes(offset, 26, 33))
        header['data_end'] = int(self.read_bytes(offset, 34, 41))
        header['analysis_start'] = int(self.read_bytes(offset, 42, 49))
        header['analysis_end'] = int(self.read_bytes(offset, 50, 57))
        
        return header
        
    
    def parse_text(self, offset, start, stop):
        """return parsed text segment of fcs file"""
        text = self.read_bytes(offset, start, stop)
        #TODO: add support for suplement text segment
        return self.parse_pairs(text)
    
    def parse_analysis(self, offset, start, stop):
        if start == stop:
            return {}
        else:
            text = self.read_bytes(offset, start, stop)
            return self.parse_pairs(text)
    
    def parse_data(self, offset, start, stop, text):
        print text['datatype']
        print text['mode']
        pass
    
    def parse_pairs(self, text):
        delim = text[0]
        if delim != text[-1]:
            warn("text in segment does not start and end with delimiter")
        tmp = text[1:-1].replace('$','')
        # match the delimited character unless it's doubled
        regex = re.compile('(?<=[^%s])%s(?!%s)' % (delim, delim, delim))
        tmp = regex.split(tmp)
        return dict(zip([ x.lower() for x in tmp[::2]], tmp[1::2]))
    
if __name__ == '__main__':
    foo = FCSreader('/home/jolly/Projects/flow/data/3FITC_4PE_004.fcs')
    print foo.parse_header(0)
    bar = foo.parse_text(0,58,1376)
    print bar.keys()
    baz = foo.parse_data(0,1337, 757928, bar)
    