# -*- coding: utf-8 -*-

#    landsat2nc
#    Copyright (C) 2020  National Centre for Earth Observation (NCEO) / Agnieszka Walenkiewicz
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import xarray as xr
from osgeo import osr, gdal
import numpy as np
import logging

class Projection:

    wgs84_wkt = """
        GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""

    def __init__(self,source_projection):
        self.source_projeciton = source_projection

        # WGS coordinate system
        wgs84_coord_sys = osr.SpatialReference()
        wgs84_coord_sys.ImportFromWkt(Projection.wgs84_wkt)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.source_projeciton)

        self.source_to_wgs84 = osr.CoordinateTransformation(srs, wgs84_coord_sys)

    def get_source_to_wgs84_transform(self):
        return self.source_to_wgs84

    @staticmethod
    def setup(source_projection):
        Projection.projector = Projection(source_projection)

class TiffImporter:

    def __init__(self):
        pass

    def latlon_image(self, path):

        # Open input dataset
        indataset = gdal.Open(path, gdal.GA_ReadOnly)
        Projection.setup(indataset.GetProjection())

        # Read geotransform matrix and calculate ground coordinates
        geomatrix = indataset.GetGeoTransform()
        pixel = indataset.RasterXSize
        line = indataset.RasterYSize

        ct = Projection.projector.get_source_to_wgs84_transform()

        latlon_im = np.zeros([2, line, pixel])

        logging.getLogger().info("Mapping pixel locations to target projection")
        pct = -1
        for i in np.arange(0, line, 1):
            new_pct = int(100*i/line)
            if new_pct > pct and new_pct % 5 == 0:
                logging.getLogger().info("%d"%new_pct + "%")
                pct = new_pct

            for j in np.arange(0, pixel, 1):

                # step 1 - apply geotransform to get the UTM coordinates of the pixel at i,j
                X = geomatrix[0] + geomatrix[1] * j + geomatrix[2] * i
                Y = geomatrix[3] + geomatrix[4] * j + geomatrix[5] * i

                # Shift to the center of the pixel
                X += geomatrix[1] / 2.0
                Y += geomatrix[5] / 2.0

                # step 2 - apply reprojection from UTM to WGS84
                (lat,lon,_) = ct.TransformPoint(X, Y)

                latlon_im[0, i, j] = lat
                latlon_im[1, i, j] = lon

        return latlon_im


if __name__ == '__main__':
    import sys
    input_path = sys.argv[1]    # path to a Landsat8 TIF file to read
    output_path = sys.argv[2]   # path to output netcdf4 file containing pixel lat/lon coordinates
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Extracting WGS84 pixel coordinates from image %s to %s" % (input_path, output_path))
    importer = TiffImporter()
    latlon_im = importer.latlon_image(input_path)
    logging.info("Extracted coordinates - shape %s" % str(latlon_im.shape))
    da = xr.DataArray(data=latlon_im, dims=("axis","i","j"),name="latlons")
    da.to_netcdf(output_path)
    logging.info("Extraction complete")

