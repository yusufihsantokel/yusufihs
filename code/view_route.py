'''Modules'''
import math
import argparse
import os
import pandas as pd
import json
import time
import numpy as np
import veroviz as vrv
import matplotlib.pyplot as plt
'''Files'''
from routing import routing
from neighborhoods import getNeighborhoods, createMapNeighborhoods
'''Get data provider API Key'''
ORS_API_KEY = os.environ['ORSKEY']

def getData():
    '''Return pd dataframes retrieved from files'''
    stops_df = pd.read_csv(f"data/google_transit/stops.txt")
    shapes_df = pd.read_csv(f"data/google_transit/shapes.txt")
    trips_df = pd.read_csv(f"data/google_transit/trips.txt")
    return stops_df, shapes_df, trips_df

def creatMapObj(origin_name: str, destination_name: str, mode: str):
    neighborhoods = getNeighborhoods(url  = "https://raw.githubusercontent.com/IE-670/bnmc/data/neighborhoods.json", 
                                    file = f"data/neighborhoods.json")
    nbhdMapObject = createMapNeighborhoods(neighborhoods, mapObject=None, addLabel=False)

    map_name = "route-"+ origin_name + "-" + destination_name + "-" + mode + ".html"
    print(map_name)
    return nbhdMapObject, map_name

def showWalking(routeMap, start: list, start_name: str, start_icon: str, end: list, end_name : str, end_icon: str):
    '''
    Add arcs walking from an origin to a destination to the map object
    e.g. start=[origin_lat, origin_lon] start_icon="home" start_name="Origin 24"
    end = [bus_start_lat,bus_start_lon] end_icon="star" end_name="Bus Stop 1234"
    '''

    mapObj, mapFile = routeMap

    nodesDF = vrv.initDataframe('nodes')
    #origin
    nodesDF = vrv.createNodesFromLocs(locs=[start],
                                      leafletIconPrefix="glyphicon",
                                      leafletIconType=start_icon,
                                      leafletColor="black",
                                      popupText=start_name,
                                      initNodes=nodesDF)
    #destination
    nodesDF = vrv.createNodesFromLocs(locs=[end],
                                      leafletIconPrefix="glyphicon",
                                      leafletIconType=end_icon,
                                      leafletColor="black",
                                      popupText=end_name,
                                      initNodes=nodesDF)

    assignmentsDF = vrv.initDataframe('assignments')
    #arcs
    shapepointsDF = vrv.getShapepoints2D(
                    startLoc         = start,
                    endLoc           = end,
                    routeType        = 'pedestrian',
                    leafletColor     = 'black',
                    dataProvider     = 'ORS-online',
                    dataProviderArgs = {'APIkey': ORS_API_KEY})
    assignmentsDF = pd.concat([assignmentsDF, shapepointsDF], ignore_index=True, sort=False)

    #add to map
    mapObj = vrv.createLeaflet(mapObject= mapObj, mapFilename= mapFile, nodes= nodesDF, arcs= assignmentsDF)

    return mapObj, mapFile


def isCloseTo(i,j,k,tolerance):
    '''
    Return True if c is on or next to the line connecting a and b
    '''
    #change lat lon order to reflect x and y
    i = [ i[1], i[0]]
    j = [ j[1], j[0]]
    k = [ k[1], k[0]]

    delta_x = j[0] - i[0]
    delta_y = j[1] - i[1]
    direction = None
   
    #Check if ab is a vertical line
    if delta_x == 0 and delta_y != 0:
        if delta_y < 0:
            direction = "down"
        elif delta_y > 0: #vertical going up
            direction = "up"
    #Check if line is horizontal
    elif delta_x != 0 and delta_y == 0:
        if delta_x < 0:
            direction = "left"
        elif delta_x > 0:
            direction = "right"
    #Check direction of diagonal line
    elif delta_x >= 0 and delta_y >= 0: #ij goes up from left to right or just straight up
        direction = "up-right"
    elif delta_x >= 0 and delta_y <= 0:
        direction = "down-right"
    elif delta_x <= 0 and delta_y >= 0:
        direction = "up-left"   
    elif delta_x <= 0 and delta_y <= 0:    
        direction = "down-left"
        
    if direction=="up":
        top_right   = [i[0] + tolerance, i[1] + delta_y]
        bottom_left = [i[0] - tolerance, i[1]]
    elif direction=="down":
        top_right   = [i[0] + tolerance, i[1]]
        bottom_left = [i[0] - tolerance, i[1] + delta_y]
    elif direction=="right":
        top_right   = [i[0] + delta_x, i[1] + tolerance]
        bottom_left = [i[0], i[1] - tolerance]
    elif direction == "left":
        top_right   = [i[0], i[1] + tolerance]
        bottom_left = [i[0] + delta_x, i[1] - tolerance]
    elif direction=="up-right":   
        top_right   = [i[0] + delta_x + tolerance, i[1] + delta_y]
        bottom_left = [i[0] - tolerance, i[1]]
    elif direction=="up-left":
        top_right   = [i[0] + tolerance, i[1] + delta_y]
        bottom_left = [i[0] + delta_x - tolerance, i[1]]
    elif direction=="down-right":
        top_right   = [i[0] + delta_x + tolerance, i[1] + delta_y]
        bottom_left = [i[0] - tolerance, i[1] + delta_y]
    elif direction=="down-left":
        top_right   = [i[0] + tolerance, i[1]]
        bottom_left = [j[0] + delta_x - tolerance, i[1] + delta_y]

    # Check if point_c is within the rectangle boundaries
    return (bottom_left[0] <= k[0] <= top_right[0] and bottom_left[1] <= k[1] <= top_right[1])

def getDistancebw(a,b):
    distance = ( ((b[0] - a[0])**2) + ((b[1] - a[1]) **2) ) ** 0.5
    return distance

def getCommuteShape(trip_shape, bus_start, bus_end):
    '''
    yktfv
    '''
    start_idx, start_pt, end_idx, end_pt = 0, [0,0], 0, [0,0]
    fig, ax = plt.subplots()
    ax.plot(bus_start[1], bus_start[0], marker='o', label='Bus stop start')
    for i in trip_shape.index:
        j = i+1
        if j <= trip_shape.index.max():
            loc_i = list([ trip_shape.iloc[i]["shape_pt_lat"] , trip_shape.iloc[i]["shape_pt_lon"] ])
            loc_j = list([ trip_shape.iloc[j]["shape_pt_lat"] , trip_shape.iloc[j]["shape_pt_lon"] ])
            plt.plot([loc_i[1], loc_j[1]], [loc_i[0], loc_j[0]])
            if isCloseTo(loc_i,loc_j,bus_start,1e-3):
                start_idx = j
                start_pt = loc_j
                break
    plt.legend()
    plt.show()

    print(f"Bus start {bus_start} and closest shape point {start_pt}")
    
    for i, _ in trip_shape.loc[start_idx:].iterrows():
        j = i+1
        if j <= trip_shape.index.max():
            loc_i = [ trip_shape.iloc[i]["shape_pt_lat"] , trip_shape.iloc[i]["shape_pt_lon"] ]
            loc_j = [ trip_shape.iloc[j]["shape_pt_lat"] , trip_shape.iloc[j]["shape_pt_lon"] ]
            if isCloseTo(loc_i,loc_j,bus_end,1e-3):
                end_idx = i
                end_pt = loc_i
                break
    
    print(f"Bus end {bus_end} and closest shape point {end_pt}")
    
    if start_idx == end_idx:
        if getDistancebw(bus_start, end_pt) >= getDistancebw(bus_start, bus_end):
            shape = None
        else:
            shape = end_pt
    else:
        shape = []
        for i,row in trip_shape.iloc[start_idx:end_idx+1].iterrows():
            lat = row["shape_pt_lat"]
            lon = row["shape_pt_lon"]
            shape.append([lat,lon])
    
    return shape #return as a np.array instead of dataframe


def showBus(routeMap, bus_start: list, bus_end: list, route_shape):
    '''
    yktv
    '''
    mapObj, mapFile = routeMap

    assignmentsDF = vrv.initDataframe('assignments')
    if route_shape:
        #Add first arc from the bus_stop_start to the first shape point
        shapepointsDF = vrv.getShapepoints2D(
                        startLoc         = bus_start,
                        endLoc           = route_shape[0],
                        routeType        = 'fastest',
                        leafletColor     = 'black',
                        dataProvider     = 'ORS-online',
                        dataProviderArgs = {'APIkey': ORS_API_KEY})
        assignmentsDF = pd.concat([assignmentsDF, shapepointsDF], ignore_index=True, sort=False)

        if len(route_shape)>1:
            for i, pt in enumerate(route_shape):
                j = i+1
                if j < len(route_shape):
                    shapepointsDF = vrv.getShapepoints2D(
                                startLoc         = route_shape[i],
                                endLoc           = route_shape[j],
                                routeType        = 'fastest',
                                leafletColor     = 'black',
                                dataProvider     = 'ORS-online',
                                dataProviderArgs = {'APIkey': ORS_API_KEY})
                    assignmentsDF = pd.concat([assignmentsDF, shapepointsDF], ignore_index=True, sort=False)

        #Add last arc from the last shape point to bus_stop_end
        shapepointsDF = vrv.getShapepoints2D(
                startLoc         = route_shape[-1],
                endLoc           = bus_end,
                routeType        = 'fastest',
                leafletColor     = 'black',
                dataProvider     = 'ORS-online',
                dataProviderArgs = {'APIkey': ORS_API_KEY})
        assignmentsDF = pd.concat([assignmentsDF, shapepointsDF], ignore_index=True, sort=False)
    
    else:
        #shape is simply a straight line connecting two bus stops
        shapepointsDF = vrv.getShapepoints2D(
                startLoc         = bus_start,
                endLoc           = bus_end,
                routeType        = 'fastest',
                leafletColor     = 'black',
                dataProvider     = 'ORS-online',
                dataProviderArgs = {'APIkey': ORS_API_KEY})
        assignmentsDF = pd.concat([assignmentsDF, shapepointsDF], ignore_index=True, sort=False)

    nodesDF = vrv.initDataframe('nodes')
    mapObj = vrv.createLeaflet(mapObject = mapObj, mapFilename=mapFile, nodes=nodesDF, arcs=assignmentsDF)

    return mapObj, mapFile


def viewRoute(origin: int, destination: int, preference):
    '''
    Visualize the result returned from experiment
    Parameters
    ----------
    result : pd.Series
        The row of the results.csv file which represents a single O/D route to be visualized
    Returns
    -------
    routeMap
        Map created using leaflet in veroviz
    '''
    routeMap = creatMapObj(origin_name, destination_name, preference)

    #TODO: get result of the OD pair from results.csv

    if preference=="walk":
        #TODO: get origin location and destination location
        # origin_loc = [origin["lat"],origin["lon"]]
        # destination_loc = [destination["lat"],destination["lon"]]
        routeMap = showWalking(routeMap, origin_loc,origin_name,"home",destination_loc,destination_name,"home")

    else:
        #get trip_id and stop-id for starting and ending bus stop
        trip_id = int(result["trip_id"])
        start_id= int(result['bus_id_start'])
        end_id = int(result['bus_id_end'])
        #get origin location and destination location
        origin_loc = [ result["origin_lat"], result["origin_lon"]]
        origin_name = result["start_name"]
        destination_loc = [ result["destination_lat"], result["destination_lon"]]
        destination_name = result["poi_name"]
        
        stops, shapes, trips = getData()
        #find location of the starting bus stop from stops_df
        bus_start_lat = lookup(stops, "stop_id", start_id, "stop_lat")
        bus_start_lon = lookup(stops, "stop_id", start_id, "stop_lon")
        bus_start_loc = [ float(bus_start_lat), float(bus_start_lon) ]
        #add walking from origin to the bus stop
        routeMap = showWalking(routeMap, origin_loc, origin_name,"home", bus_start_loc, "Bus Stop" + str(start_id), "star")
        #find location of the ending bus stop from stops df
        bus_end_lat = lookup(stops, "stop_id", end_id, "stop_lat")
        bus_end_lon = lookup(stops, "stop_id", end_id, "stop_lon")
        bus_end_loc = [ float(bus_end_lat), float(bus_end_lon) ]
        #add walking from bus stop to destination
        routeMap = showWalking(routeMap, bus_end_loc, "Bus Stop" + str(end_id), "star", destination_loc, destination_name, "home")

        #Get the shape_id for this trip from trips df
        shape_id = lookup(trips, "trip_id", trip_id, "shape_id")
        #Lookup that shape_id in shapes to get the rows representing the shape of this trip
        trip_shape = shapes.loc[shapes["shape_id"] == shape_id]
        #sort the rows by stop sequence
        trip_shape = pd.DataFrame(trip_shape).sort_values(by="shape_pt_sequence").reset_index(drop=True)
        #get the points in the shape of the commuter's trip only between the bus stops
        commute_shape = getCommuteShape(trip_shape, bus_start_loc, bus_end_loc)
        #plot the shape of the trip betwen the starting and ending bus stops
        routeMap = showBus(routeMap, bus_start_loc, bus_end_loc, commute_shape)

    return routeMap

if __name__ == '__main__':

