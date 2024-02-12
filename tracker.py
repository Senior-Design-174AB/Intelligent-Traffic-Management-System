import cv2
import numpy as np
import argparse
import heapq

from scipy.spatial import distance as dist

PATH_FRAME = 20

class Tracker:
    def __init__(self, maxLost=30, maxDistance=50):
        self.nextObjectID = 0
        self.deregisteredID = []
        self.objects = {}
        self.lost = {}
        self.paths = {}
        self.maxLost = maxLost
        self.maxDistance = maxDistance

    def register(self, centroid):
        if self.deregisteredID:
            nextID = heapq.heappop(self.deregisteredID)
        else:
            nextID = self.nextObjectID
            self.nextObjectID += 1
        
        self.objects[nextID] = centroid
        self.lost[nextID] = 0
        self.paths[nextID] = [centroid]

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.lost[objectID]
        del self.paths[objectID]
        heapq.heappush(self.deregisteredID, objectID)

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.lost.keys()):
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = self._distance(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.lost[objectID] = 0
                self.paths[objectID].append(inputCentroids[col])
                
                if len(self.paths[objectID]) > PATH_FRAME:
                    self.paths[objectID].pop(0)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.lost[objectID] += 1

                if self.lost[objectID] > self.maxLost:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

    def get_paths(self):
        return self.paths

    def _distance(self, a, b):
        return dist.cdist(a, b, metric='euclidean')
