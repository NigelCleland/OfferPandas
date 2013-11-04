#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import defaultdict
import datetime

class OfferFrame(DataFrame):
    """docstring for OfferFrame"""


    def __new__(cls, *args, **kargs):

        arr = DataFrame.__new__(cls, *args, **kargs)

        if type(arr) is OfferFrame:
            return arr
        else:
            return arr.view(OfferFrame)

    def transmogrify(self):
        arr = self._retitle_columns()._map_frame()._convert_date()
        arr = arr._create_timestamp().stack_frame()._market_node()
        return arr

    def _retitle_columns(self):

        self.rename(columns={x: x.strip().title()
                    for x in self.columns}, inplace=True)

        grid_names = ("Grid_Point", "Grid_Injection_Point", "Grid_Exit_Point")
        self.rename(columns={x: "Node" for x in grid_names}, inplace=True)

        return self


    def _map_frame(self):
        mapping = pd.read_csv("_static/nodal_metadata.csv")
        return OfferFrame(self.merge(mapping, left_on="Node", right_on="Node"))


    def stack_frame(self):
        arr = pd.concat(self._stack_frame(), ignore_index=True)
        max_names = ("Power", "Max")
        arr.rename(columns={x: "Quantity" for x in max_names}, inplace=True)
        return OfferFrame(arr)


    def _stack_frame(self):

        general_columns = [x for x in self.columns if  "Band" not in x]
        fdict = self._classify_bands()

        for key in fdict:
            allcols = general_columns + fdict[key].values()
            single = self[allcols].copy()
            single["Product_Type"] = key[0]
            single["Reserve_Type"] = key[1]
            single["Band"] = key[2]
            single.rename(columns={v:k for k, v in fdict[key].items()},
                          inplace=True)
            yield single


    def _classify_bands(self):
        band_columns = [x for x in self.columns if "Band" in x]
        fdict = defaultdict(dict)
        for band in band_columns:
            number, param = (int(band.split('_')[0][4:]), band.split('_')[-1])
            rt, pt = (self._reserve_type(band), self._product_type(band))
            fdict[(pt, rt, number)][param] = band
        return fdict



    def _reserve_type(self, band):
        return "FIR" if "6S" in band else "SIR" if "60S" in band else "Energy"


    def _product_type(self, band):
        if "Plsr" in band:
            return "PLSR"
        elif "Twdsr" in band:
            return "TWDSR"
        elif "6S" in band or "60S" in band:
            return "IL"
        else:
            return "Energy"


    def _convert_date(self):
        self["Trading_Date"] = pd.to_datetime(self["Trading_Date"])
        return self


    def _create_timestamp(self):
        num_min = {x: datetime.timedelta(minutes=x*30-15)
                  for x in self["Trading_Period"].unique()}
        self["Timestamp"] = self["Trading_Date"] + self["Trading_Period"].map(num_min)
        return self

    def efilter(self, **kargs):
        arr = self.copy()
        for key, value in kargs.iteritems():
            arr = arr[arr[key] == value]
        return OfferFrame(arr)

    def rfilter(self, **kargs):
        arr = self.copy()
        for key, values in kargs.iteritems():
            arr = arr[(arr[key] >= values[0]) & (arr[key] <= values[1])]
        return OfferFrame(arr)

    def nfilter(self, **kargs):
        arr = self.copy()
        for key, value in kargs.iteritems():
            self = arr[arr[key] != value]

        return OfferFrame(self)

    def price_stack(self):
        return self.groupby(["Timestamp", "Price"])["Quantity"].sum()

    def incrementalise(self):
        self["Incr Quantity"] = 1
        return pd.concat(self._single_increment(), axis=1).T

    def _market_node(self):
        lammn = lambda x: " ".join([x[0], "".join([x[1], str(x[2])])])
        self["Market_Node_ID"] = self[["Node", "Station", "Unit"]].apply(
                        lammn, axis=1)
        return self

    def _single_increment(self):
        """ Note assume a single timestamp """
        for index, series in self.iterrows():
            power = series["Quantity"]
            while power > 0:
                series["Incr Quantity"] = min(1, power)
                yield series
                power -= 1
