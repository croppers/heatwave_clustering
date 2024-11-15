import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gc
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

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
    This function identifies heatwaves based on the exceedance threshold, duration,
    and the condition that intervals between two events must be six days or longer.

    Parameters
    ----------
    timeseries: xarray DataArray or Dataset
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
    heatwave_events: list of lists
        A list containing heatwave events, where each event is a list of datetime objects
        representing consecutive days exceeding the temperature threshold.
    '''
    # Compute the threshold temperature
    threshold = timeseries.quantile(exceedance, dim='day')

    # Define a mask of the days when the temperature exceeds the threshold
    exceeds = timeseries > threshold

    # Convert the mask to a Pandas Series and extract days exceeding the threshold
    exceeds_series = exceeds.to_series().reset_index()
    data_column = exceeds_series.columns[-1]
    exceeds_true = exceeds_series[exceeds_series[data_column]]
    days = exceeds_true['day'].sort_values()

    # Convert 'days' to datetime
    if not np.issubdtype(days.dtype, np.datetime64):
        days = pd.to_datetime(days.astype(int).astype(str), format='%Y%m%d', errors='coerce')
    days = days.dropna()

    # Identify events where the gap between days is six days or more
    day_diffs = days.diff().dt.days
    new_event = day_diffs >= 6  # Start a new event if gap is 6 days or more
    event_ids = new_event.cumsum()
    events = days.groupby(event_ids)

    # Collect heatwave events
    heatwave_events = []
    for event_id, event_days in events:
        # Within each event, identify sequences of consecutive days
        event_days = event_days.sort_values()
        day_diffs = event_days.diff().dt.days
        new_group = day_diffs != 1  # Start a new group if days are not consecutive
        group_ids = new_group.cumsum()
        groups = event_days.groupby(group_ids)
        for group_id, group_days in groups:
            if len(group_days) >= duration:
                heatwave_events.append(list(group_days.values))

    return heatwave_events

def get_features(phi_ds, events, pca=True):
    '''
    This function extracts features from the heatwave events.

    Parameters:
    -----------
    events: list of lists
        A list containing heatwave events, where each event is a list of datetime objects
        representing consecutive days exceeding the temperature threshold.
    phi_da: xarray DataArray
        The geopotential height data.
        Dimensions - (day, lat, lon)
    pca: bool
        Whether to apply PCA to the feature vectors.
        Default value is True.
    '''
    # compute the long-term daily mean
    long_term_daily_mean = phi_ds.groupby('time.dayofyear').mean(dim='time')

    # smooth the long-term daily mean with a fourier transform
    data = long_term_daily_mean['phi_3d'].values  # shape (dayofyear, lat, lon)
    day_len = data.shape[0]
    fft_coeffs = np.fft.rfft(data, axis=0)     # perform rFFT along 'dayofyear' dimension
    fft_coeffs[4+1:] = 0    # zero out all but the first 4 harmonics
    data_smoothed = np.fft.irfft(fft_coeffs, n=day_len, axis=0) # Inverse rFFT to get smoothed data
    smoothed_daily_mean = xr.DataArray(
        data_smoothed, dims=long_term_daily_mean.dims, coords=long_term_daily_mean.coords
        )

    # compute anomalies by subtracting the smoothed long-term daily mean from the total field
    anomalies = phi_ds.groupby('time.dayofyear') - smoothed_daily_mean
    anomalies = anomalies.rename({'time': 'time'})

    # prepare features for clustering using anomalies
    features = []
    event_ids_ordered = []  # list to store event IDs in the same order as features

    for event in events:
        event_days = [event[0] + pd.Timedelta(days=i) for i in range(-3, 4)]
        feature_vectors = []
        incomplete_event = False
        for day in event_days:
            day_pd = pd.to_datetime(day)

            # select the anomaly data for the specific day
            try:
                ds_day_anomaly = anomalies.sel(time=day_pd)
            except KeyError:
                print(f"Anomaly data not found for day {day_pd.strftime('%Y-%m-%d')}")
                incomplete_event = True
                break  # skip events that are in between files

            # flatten the data
            anomaly_flat = ds_day_anomaly['phi_3d'].values.flatten()
            feature_vectors.append(anomaly_flat)

        if not incomplete_event and len(feature_vectors) == 3:
            # heatwave_feature = np.mean(feature_vectors, axis=0)
            heatwave_feature = np.array(feature_vectors).flatten()
            features.append(heatwave_feature)
            
            # store the event start date as the event ID
            event_start_date = pd.to_datetime(event[0]).strftime('%Y-%m-%d')
            event_ids_ordered.append(event_start_date)
        elif incomplete_event:
            print(f"Skipped incomplete heatwave event starting on {pd.to_datetime(event[0]).strftime('%Y-%m-%d')}")
        else:
            print(f"Unexpected condition for heatwave event starting on {pd.to_datetime(event[0]).strftime('%Y-%m-%d')}")

        # clean up to free memory
        del event_days, feature_vectors
        gc.collect()

    del events
    gc.collect()

    features_array = np.nan_to_num(np.array(features, dtype='float32'), nan=0)

    if pca:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        pca = PCA(n_components=15, random_state=0)
        return pca.fit_transform(features_scaled)
    else:
        return np.nan_to_num(np.array(features, dtype='float32'), nan=0)

def get_clusters(region, T_da, var, phi_da, k=2, pca=True):
    '''
    This function performs clustering on the heatwave events.

    Parameters
    ----------
    region: List[float]
        The bounding box of the region of interest.
        Order: [lat_lb, lat_ub, lon_lb, lon_ub]
    T_da: xarray DataArray
        The temperature data.
        Dimensions - (day, lat, lon)
    var: str
        The variable of interest (e.g., 'tmax', 'tmin', 'tmean').
    phi_da: xarray DataArray
        The geopotential height data.
        Dimensions - (day, lat, lon)
    k: int
        The number of clusters to create.
    '''
    T_ts = get_temperature_timeseries(T_da, var, region[0], region[1], region[2], region[3])
    events = get_heatwaves(T_ts)
    features = get_features(phi_da, events, pca=pca)

    km = KMeans(n_clusters=k, random_state=0)
    km.fit(features)
    return km