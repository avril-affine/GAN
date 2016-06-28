##############################################################################
#
# Functions to use the flicker api to search and download images.
#
# Modified from https://github.com/k-lev/City_Tagger
#
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import urllib, urlparse
import flickrapi
import datetime as dt
import pprint
import os.path
import time


def get_flickr_keys():
    '''
    Read flckr key and secret key from .json file
    Return flckr key and secret key.
    '''
    # Get skey and ecret key for the flikr api
    with open('/Users/PANDA/flickr_key.json') as f:
        data = json.load(f)
        api_key = data['Key']
        api_secret = data['Secret']

    return api_key, api_secret

def flkr_search(type, tags):
    '''
    Input:  list of tags
    Search Flckr for all images tagged with list 'tags'.
    Download and save all images not already saved.
    '''
    api_key, api_secret = get_flickr_keys()

    # create Flickerapi instance
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

    output = 'flickr/'
    prefix = "http://farm{farm}.staticflickr.com/{server}/"
    suffix = "{id}_{secret}_b.jpg"
    template = prefix + suffix

    # Set the first day as today; date range per search = day
    end_time = dt.datetime.now()
    day = dt.timedelta(days=28)
    start_time = end_time-day

    count = 0

    #download until I have 5000
    # Actual total for chicago skyline:  2437 (including junk)
    while count < 10000:
        #find how many pages in this date range
        if type == 'tags':
            results = flickr.photos.search(tags=tags,
                                           tag_mode='all',
                                           extras='autotags',
                                           page=1,
                                           min_upload_date=start_time,
                                           max_upload_date=end_time,
                                           per_page=500)
            total_pages = results['photos']['pages']
        else:
            results = flickr.photos.search(text=tags,
                                           tag_mode='all',
                                           extras='autotags',
                                           page=1,
                                           min_upload_date=start_time,
                                           max_upload_date=end_time,
                                           per_page=500)
            total_pages = results['photos']['pages']

        print '************ '+start_time.strftime("%Y%m%d") +' **************'

        # loop through searches, one for each page in this date range
        for page in xrange(1, total_pages+1):
            if type == 'tags':
                results = flickr.photos.search(tags=tags,
                                               tag_mode='all',
                                               extras='autotags',
                                               page=page,
                                               min_upload_date=start_time,
                                               max_upload_date=end_time,
                                               per_page=500)
                total_pages = results['photos']['pages']
            else:
                results = flickr.photos.search(text=tags,
                                               tag_mode='all',
                                               extras='autotags',
                                               page=page,
                                               min_upload_date=start_time,
                                               max_upload_date=end_time,
                                               per_page=500)
                total_pages = results['photos']['pages']

            #for each result, download the image if it does not alreay exist in the directory
            for i, photo in enumerate(results['photos']['photo']):
                url = template.format(**photo)
        #         index = "%0.6i" % (i+250*(page - 1)
        #         local = output + index + "-" + suffix.format(**photo)
                local = output + suffix.format(**photo)
                if os.path.exists(local) == False:
                    count += 1
                    print 'total: ', count
                    print "* saving", url
                    urllib.urlretrieve(url, local)
                    print "      as", local
                    time.sleep(.1)

        #update the date range
        end_time = start_time
        start_time = start_time-day
        

if __name__ == '__main__':
    flkr_search('tag', ['cute puppy'])
    flkr_search('text', ['cute puppy'])
