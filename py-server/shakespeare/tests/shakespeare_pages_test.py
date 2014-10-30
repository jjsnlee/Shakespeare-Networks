from unittest import TestCase, main
import shakespeare.shakespeare_pages as sp

class ShakespearePagesTest(TestCase):
    
    def test_get_play_data_json(self):
        req = None 
        play_set = None
        rsp = sp.get_play_data_json(req, play_set)
        json_rslt = rsp.content
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    main()