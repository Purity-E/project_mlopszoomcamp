# import batch file
import batch

# testing for normalize_document() function

def test_normalize_document():
    data = {'title': 'New Poll Shows Clinton Crushing Trump In Key Swing States'}
     # defining actual
    actual_doc = batch.normalize_document(data['title'])
     #defining expected
    expected_doc = 'new poll shows clinton crushing trump in key swing states'

     #comparing expected vs actual
    assert actual_doc == expected_doc

# testing for prepare_feature() function

def test_prepare_feature():
    data = {'title': 'New Poll Shows Clinton Crushing Trump In Key Swing States'}
     # defining actual
    actual_doc = batch.prepare_feature(data['title'])
    #defining expected
    expected_doc = ['new poll show clinton crush trump key swing state']
    #comparing expected vs actual
    assert actual_doc == expected_doc