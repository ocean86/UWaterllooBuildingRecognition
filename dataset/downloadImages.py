# This script is for the collection of training & testing data for Machine Learning project
# Using google maps API, this script will take street view images around a building

import math
import requests
import numpy as np
import random

# Free Key, max 25,000 maps / 24hrs, max 640 * 640
key = 'xxx'
baseUrl = 'https://maps.googleapis.com/maps/api/streetview?size={}&location={}&heading={}&pitch={}&fov={}&key={}'

#######
# set these parameters
size = '600x400'
building = 'dc' # {dc, mc, m3}
centre = (43.472453, -80.542026) #Centre position of the building / The position camera is facing to
cam1 = (43.473572, -80.541930) # The first camera location
cam2 = (43.472949, -80.543741) # The last camera location
stops = 25 # Number of camera location of alone the line between cam1 and cam2
runName = 'a' # a character for naming image generated for this run.

pitch_default = 10 # default is 10 degrees above horizontal, random range: 10 - 35
pitch_min = -5
pitch_max = 20
fov_default = 90 # default is max zoom out, random range 40 - 120
fov_min = 30
fov_max = 100
#######

# Compute p1's heading towards p2
def computeHeading(p1, p2):
    # pdb.set_trace()
    # https://gist.github.com/jeromer/2005586
    lat1 = math.radians(p1[0])
    lat2 = math.radians(p2[0])
    diffLong = math.radians(p2[1] - p1[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    heading = math.atan2(x, y)
    heading = (math.degrees(heading) + 360) % 360
    return str(heading) # Return String incase of lost of precision

# Download a image Through Google Maps API
def requestImage(size, location, heading, pitch, fov, key, random, fileno):
    req = baseUrl.format(size, location, heading, pitch, fov, key)

    # print(req)
    res = requests.get(req)

    if res.status_code != requests.codes.ok:
        print('request failed. {}'.format(res.status_code))
    else:
        filename = building + '/' + runName + '_' + str(fileno) + '.jpg'
        if random:
            filename = building + '/' + runName + '_' + 'random_' + str(fileno) + '.jpg'
        with open(filename, 'wb') as fd:
            for chunk in res.iter_content(chunk_size=128):
                fd.write(chunk)
        print(filename)

# Build camera points:
xList = np.linspace(cam1[0], cam2[0], stops)
yList = np.linspace(cam1[1], cam2[1], stops)
locations = []
for i in range(stops):
    locations.append((xList[i], yList[i]))

# Get images
fileno = 1
for p in locations:
    heading = str(computeHeading(p, centre))
    location = str(p[0]) + ',' + str(p[1])
    randomPitch = random.randint(pitch_min, pitch_max)
    randomFov = random.randint(fov_min, fov_max)
    requestImage(size, location, heading, pitch_default, fov_default, key, False, fileno)
    requestImage(size, location, heading, randomPitch, randomFov, key, True, fileno)
    fileno += 1
