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

    def retitle_columns(self):

        self.rename(columns={x: x.strip().replace('_', ' ').title()
                    for x in self.columns}, inplace=True)

        grid_names = ("Grid Point", "Grid Injection Point", "Grid Exit Point")
        self.rename(columns={x: "Node" for x in grid_names}, inplace=True)

        return self

    def map_frame(self):
        mapping = pd.read_csv("_static/nodal_metadata.csv")
        return OfferFrame(self.merge(mapping, left_on="Node", right_on="Node"))

    def stack_frame(self):
        return OfferFrame(pd.concat(self._stack_frame(), ignore_index=True))

    def _stack_frame(self):

        general_columns = [x for x in self.columns if  "Band" not in x]
        fdict = self._classify_bands()

        for key in fdict:
            allcols = general_columns + fdict[key].values()
            single = self[allcols].copy()
            single["Product Type"] = key[0]
            single["Reserve Type"] = key[1]
            single["Band"] = key[2]
            single.rename(columns={v:k for k, v in fdict[key].items()},
                          inplace=True)
            yield single

    def _classify_bands(self):
        band_columns = [x for x in self.columns if "Band" in x]
        fdict = defaultdict(dict)
        for band in band_columns:
            number, param = (int(band.split()[0][4:]), band.split()[-1])
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
        self["Trading Date"] = pd.to_datetime(self["Trading Date"])
        return self

    def _create_timestamp(self):
        num_min = {x: datetime.timedelta(minutes=x*30-15)
                  for x in self["Trading Period"].unique()}
        self["Timestamp"] = self["Trading Date"] + self["Trading Period"].map(num_min)
        return self


