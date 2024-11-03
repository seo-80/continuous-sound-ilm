# test xarray
import xarray as xr
import numpy as np

# Create a DataArray
data = np.random.rand(4, 3)
coords = {'x': ['a','a','c','d'], 'y': np.arange(3)}
dims = ('x', 'y')
da = xr.DataArray(data, coords=coords, dims=dims)
print(da)
print(da.sel(x='a').values)
print(da.a)