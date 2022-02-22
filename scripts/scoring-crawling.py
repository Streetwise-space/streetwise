#!/bin/python3
# -*- coding: utf-8 -*-
from dotenv import load_dotenv

import mapillary.interface as mly

import os
import cv2
import wget
import json
import libgeohash as gh
import datetime
import image_processing as imgp
import pandas as pd

load_dotenv()

##*****************Configuration ***************

OUTPUT_FOLDER = "../data/"

key = os.getenv("MAPILLARY_KEY") # get access key from environment variable

gh_cantons={
  "SG":{"area_gh":"u0qt","suffix":["v3","v7","tx","ub","vu","su"]},
  # "ZH":{"area_gh":"u0qj", "suffix":["d"]}
}

# We get images from 2015 onwards
start_date = datetime.datetime(2015,1,1)

# In this resolution (256, 1024, and 2048 possible)
image_resolution = 1024


def start_crawl(areas):
  # Used to name images
  i = 0
  for key_canton in areas.keys():
      area_canton=0
      while area_canton < len(gh_cantons[key_canton]["suffix"]):
          gh_urban = gh_cantons[key_canton]["area_gh"] + gh_cantons[key_canton]["suffix"][area_canton]
          #gh_urban = geohash_compl[area_canton]
          print("Geohash query: %s" % gh_urban)
          
          #gh_box_map=convertBBoxtoMapillary(gh_urban)
          #we query sequences because we don't want many similar images in one area
          # no limit is set as we want to query all images.
          #raw_json = map.search_sequences(bbox=gh_box_map, start_time=start_date)
          
          raw_data = mly.image.get_images_in_bbox_controller(
            bounding_box = getMapillaryBbox(gh_urban), 
            layer = 'image',
            zoom = 14,
            # is_image = True,
            filters = {
              'image_type': 'flat',
              #'min_captured_at': start_date,
            }
          )
          raw_json = json.loads(raw_data)
          #print(raw_json)

          # every feature is a sequence of pictures
          sequence_list = raw_json["features"]
          print("Downloading %d features" % len(sequence_list))

          # TODO: roll up into a single entry
          for feature in sequence_list:
            if str(feature["properties"]["is_pano"]) != "False": 
              # Skip panoramas
              continue
            image_keys = [feature["properties"]["id"]]
            coordinates = feature["geometry"]["coordinates"]
            image_angles = feature["properties"]["compass_angle"]
            sequence_info = ','.join([
              str(feature["properties"]["sequence_id"]),
              str(feature["properties"]["captured_at"]),
              str(feature["properties"]["is_pano"])
            ])
            i = register_entry(image_keys, coordinates, key_canton, i, image_resolution, image_angles, sequence_info)

          area_canton += 1


def convertBBoxtoMapillary(geohash):
    """
    Converts a geohash into a string  representing the bounding box coordinates in this order:
        Args:
            geohash (string):Geohash whose bounding box we want to obtain e.g. u0qj9

        Return:
            (string): bounding box  coordinates string in format West,South,East,North
    """
    gh_bbox = gh.bbox(geohash)
    return str(gh_bbox["w"])+","+str(gh_bbox["s"])+","+str(gh_bbox["e"])+","+str(gh_bbox["n"])


def getMapillaryBbox(geohash):
    """
    Converts a geohash into a string  representing the bounding box coordinates in this order:
        Args:
            geohash (string):Geohash whose bounding box we want to obtain e.g. u0qj9

        Return:
            (dict): a bounding box as a dictionary
    """
    gh_bbox = gh.bbox(geohash)
    return {
        "west":  gh_bbox["w"],
        "south": gh_bbox["s"],
        "east":  gh_bbox["e"],
        "north": gh_bbox["n"]
    }



def register_entry(image_keys, coord_list, key_canton, i, ir, image_angles, sequence_info):
    """
    Downloads an image from mapillary given the image key and registers the entry in an existing  csv file.

           Args:
               image_keys ([int]): an array of Mapillary image keys (from a sequence)
               coord_list (list): list of coordinates corresponding to the images sent in image_keys
               key_canton (string): represents the canton where the image belongs, to be registered in the CSV file
               i (int): index to be added to the image name
               ir (int): image resolution
               image_angles (list): angles in which the images in image_keys were taken
               sequence_info (string): string containing  the mapillary key of the sequence, date when it was created
               and if it was taken as a panoramic image. This info is added to the csv for each image.
           Return:
               (int): the index used in the image name for the  images coming after this fucntion is executed

    """
    datapath = os.path.join(OUTPUT_FOLDER, "scoring.csv")
    index_image = 0
    while index_image < len(image_keys):
        filepath = os.path.join(OUTPUT_FOLDER, str(image_keys[index_image]) + ".jpg")
        flag = download_image_by_key(image_keys[index_image], ir, filepath)
        if flag:
            myfile = open(datapath, "a")
            image_name = str(image_keys[index_image]) + ".jpg"
            line=",".join([
                str(image_keys[index_image]), 
                key_canton, 
                str(coord_list[1]),
                str(coord_list[0]), 
                str(image_angles), 
                sequence_info+"\n"
            ])
            myfile.write(line)
            myfile.close()
            print("Saved %s" % image_name)
        index_image += 1
        i += 1

    return i



# https://www.mapillary.com/developer/api-documentation/#retrieve-image-sources
def download_image_by_key(key, image_resolution=1024, download_path=None):

    """Download a image by the key

    Args:
        key (string): Image key of the image you want to download.
        image_resolution (int): Resolution of the image you want to download.
        download_path (string): The download path of the file to download.

    Return:
        (boolean): True if the download is sucessful (for now)

    """
    if os.path.isfile(download_path):
        return True
    
    #try:            
    url = mly.image.get_image_thumbnail_controller(
      image_id = key, 
      resolution = image_resolution
    )
    #except:
    #    return False
    
    # Use the wget library to download the url
    print("Downloading ... %s" % key)
    filename = wget.download(url, download_path)
    return True




##*****************Main program ***************

# Create a Mapillary Object from keys in the environment
mly.set_access_token(key)

start_crawl(gh_cantons)
print("Map crawl complete.")
