from configurations import ds_config
from util import dataset, plotter
data_path = '/Volumes/Elements/ADI/data_tm20/'

# Data creation and load
ds = dataset.Dataset()
# testing testing the route
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'test_routing_config')

# see if we can access the server
ds.dataset_creation_influx()


print("yes yes")
