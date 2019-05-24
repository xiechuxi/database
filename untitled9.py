#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:48:28 2019

@author: xiechuxi
"""

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qgis.core import *

coverage = "/Users/joellawhead/qgis_data/atlas/grid.shp"
atlasPattern = "/Users/joellawhead/qgis_data/atlas/output_"

# Load the map layer. This example uses a shapefile
# but you can use any supported QGIS layer.
vlyr = QgsVectorLayer(coverage, "grid", "ogr")
QgsMapLayerRegistry.instance().addMapLayer(vlyr)

# Set up the map renderer
mr = QgsMapRenderer()
mr.setLayerSet([vlyr.id()])
mr.setProjectionsEnabled(True)
mr.setMapUnits(QGis.DecimalDegrees)
crs = QgsCoordinateReferenceSystem()
crs.createFromSrid(4326)
mr.setDestinationCrs(crs)

# Create a composition object which
# handles layouts and pages
c = QgsComposition(mr)
c.setPaperSize(297, 210)

# Set up the symbology for the shapefile.
# Not necessary for a WMS
gray = { "color": "155,155,155" }
mapSym = QgsFillSymbolV2.createSimple(gray)
renderer = QgsSingleSymbolRendererV2(mapSym)
vlyr.setRendererV2(renderer)

# Put the composer in "atlas" mode to
# zoom to features automatically.
atlasMap = QgsComposerMap(c, 20, 20, 130, 130)
atlasMap.setFrameEnabled(True)
c.addComposerMap(atlasMap)

# Configure the atlas
atlas = c.atlasComposition()
atlas.setCoverageLayer(vlyr)
atlas.setHideCoverage(False)
atlas.setEnabled(True)
c.setAtlasMode(QgsComposition.ExportAtlas)

# Optional overview map on each image
ov = QgsComposerMap(c, 180, 20, 50, 50)
ov.setFrameEnabled(True)
ov.setOverviewFrameMap(atlasMap.id())
c.addComposerMap(ov)
rect = QgsRectangle(vlyr.extent())
ov.setNewExtent(rect)

# Yellow extent box for overview map
yellow = { "color": "255,255,0,255" }
ovSym = QgsFillSymbolV2.createSimple(yellow)
ov.setOverviewFrameMapSymbol(ovSym)

# Label the map image with an attribute column
lbl = QgsComposerLabel(c)
c.addComposerLabel(lbl)
lbl.setText("[% \"GRID_ID\" %]")
lbl.setFont(QgsFontUtils.getStandardTestFont())
lbl.adjustSizeToText()
lbl.setSceneRect(QRectF(150, 5, 60, 15))

# Some more page composition info
atlasMap.setAtlasDriven(True)
atlasMap.setAtlasScalingMode(QgsComposerMap.Auto)
atlasMap.setAtlasMargin(0.10)

# Loop through each feature to zoom and create an image.
atlas.setFilenamePattern("'%s' || $feature" % atlasPattern)
atlas.beginRender()
for i in range(0, atlas.numFeatures()):
    atlas.prepareForFeature(i)
    filename = atlas.currentFilename() + ".png"
    filenames.append(filename)
    img = c.printPageAsRaster(0)
    img.save(filename, 'png')

atlas.endRender()




kf=KFold(5) 
"""Fold of 5 indicates 1-4 split 0.2 testing set"""
fold=0

for train_index,validate_index in kf.split(df):
    fold+=1
    print("Fold #{}".format(fold))
    trainDF =pd.DataFrame(df.iloc[train_index])
    validateDF=pd.DataFrame(df.iloc[validate_index])
    
    """load images"""
   
    
    

    
    for k in tqdm(validateDF['filename']):
        img1 = image.load_img(k,target_size=(224,224,3), grayscale=False)
        img1 = image.img_to_array(img1)
        test_image.append(img1)
    
    test_image=np.array(test_image)
    test_image= test_image.astype('float32')/255
    test_output=category[validate_index]
    
    
    
    
model= CapsNet(input_shape=x_train.shape[1:],n_class=,routings=10)

loaddata_andtrain('/Users/xiechuxi/Desktop/EO_rank_data.csv',20,0.2,10)
