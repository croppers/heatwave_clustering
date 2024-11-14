import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd

def get_temperature_timeseries(data, var, lat_lb, lat_ub, lon_lb, lon_ub):
    '''
    This function extracts the daily temperature timeseries (in Fahrenheit) for a given region of interest.
    It can be used for maximum, minimum, or mean temperature data.

    Parameters
    ----------
    data: xarray dataset
        The dataset containing the temperature data.
        Dimensions - (lat, lon, day)
    var: str
        The variable of interest (e.g., 'tmax', 'tmin', 'tmean').
    lat_lb: float
        The lower bound of the latitude of the region of interest.
    lat_ub: float
        The upper bound of the latitude of the region of interest.
    lon_lb: float
        The lower bound of the longitude of the region of interest.
    lon_ub: float
        The upper bound of the longitude of the region of interest.

    Returns
    -------
    timeseries_region: xarray dataset
        The daily temperature timeseries for the region of interest.
        Dimension - (day)
    '''
    # define the region of interest as a Polygon object
    region = Polygon([(lon_lb, lat_lb), (lon_ub, lat_lb), (lon_ub, lat_ub), (lon_lb, lat_ub)])

    # create the region mask and regrid it to the target grid
    mask_region = region.mask(data[var])

    # compute area weights based on latitude
    weights = np.cos(np.deg2rad(data['lat']))
    weights = weights / weights.sum()

    # compute the spatially-weighted average across the region
    data_region = data.where(mask_region)
    weights_region = weights.where(mask_region).fillna(0)
    weights_region_normalized = weights_region / weights_region.sum()
    timeseries_region = data_region.weighted(weights_region_normalized).mean(dim=['lat', 'lon'])

    return timeseries_region

def get_heatwaves(timeseries, exceedance=0.99, duration=3):
    '''
    This function identifies heatwaves based on the exceedance threshold and duration.

    Parameters
    ----------
    timeseries: xarray dataset
        The daily temperature timeseries.
        Dimension - (day)
    exceedance: float
        The threshold temperature percentile (0-1) above which a heatwave is defined.
        Default value is 0.99 (99th percentile).
    duration: int
        The minimum number of consecutive days above the threshold to define a heatwave.
        Default value is 3 days.

    Returns
    -------

    '''
    # compute the threshold temperature
    threshold = timeseries.quantile(exceedance, dim='day')

    # define a mask of the days when the temperature exceeds the threshold
    exceeds = timeseries > threshold

    # 
    exceeds_series = exceeds.to_series().reset_index()
    data_column = exceeds_series.columns[-1]
    exceeds_true = exceeds_series[exceeds_series[data_column]]
    days = exceeds_true['day'].sort_values()

    # convert 'days' to a datetime
    if not np.issubdtype(days.dtype, np.datetime64):
        days = pd.to_datetime(days.astype(int).astype(str), format='%Y%m%d', errors='coerce')
    days = days.dropna()

    # identify consecutive days
    day_diffs = days.diff().dt.days
    new_group = day_diffs != 1
    group_ids = new_group.cumsum()
    groups = days.groupby(group_ids)
    
    # collect heatwave events (consecutive days exceeding the duration)
    heatwave_events = []
    for group_id, group_days in groups:
        if len(group_days) >= duration:
            heatwave_events.append(list(group_days.values))
    
    return heatwave_events

