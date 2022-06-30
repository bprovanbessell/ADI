from models import convolutional_vae
from configurations import ds_config
from util import dataset
import best_model_params
import os
import talos
import pickle

data_path = '/Users/andreavisentin/ADI/data_tm/'
# 3 GPUS available on the cluster, we should use number 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# experiment_name = 'All_measurements_sept_oct_gcl_error'

# we want to train models for vibration, current, flux, and then all channels
experiment_name = "flux_oct_18_gcl_error"

# Data creation and load
# Make new dataset configuration for all of these tests
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)
# ds = ds.data_load(ds.name)
ds.dataset_creation_influx()
ds.data_summary()

# model_path = "saved_models"
model_path = "flux_final_model/"

if not os.path.exists(model_path):
    os.mkdir(model_path)

model = convolutional_vae.ConvolutionalVAE(model_path=model_path)
model.name = experiment_name
t = talos.Scan(x=ds.X_train,
               y=ds.X_train,
               model=model.training,
               experiment_name="vae_param_experiment",
               params=model.parameter_list,
               # params=best_model_params.flux,
               # round_limit=100,
               print_params=True)

if not os.path.exists("talos_results/"):
    os.mkdir("talos_results")

filehandler = open("talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)


"""(14999, 1)
2027
Final dataset shape:  (2027, 15000, 1)
Test size  (2027, 15000, 1)
Data normalization
Training (2096, 15000, 1) Testing (2027, 15000, 1)
Traceback (most recent call last):
  File "talos_training_all.py", line 34, in <module>
    t = talos.Scan(x=ds.X_train,
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/talos/scan/Scan.py", line 196, in __init__
    scan_run(self)
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/talos/scan/scan_run.py", line 9, in scan_run
    self = scan_prepare(self)
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/talos/scan/scan_prepare.py", line 24, in scan_prepare
    self.param_object = ParamSpace(params=self.params,
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/talos/parameters/ParamSpace.py", line 41, in __init__
    self.param_index = self._param_apply_limits()
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/talos/parameters/ParamSpace.py", line 97, in _param_apply_limits
    return sample_reducer(self.round_limit,
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/talos/reducers/sample_reducer.py", line 55, in sample_reducer
    out = r.uniform_mersenne()
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/site-packages/chances/methods.py", line 31, in uniform_mersenne
    return random.sample(range(self.len), k=self.n)
  File "/home/bprovan/anaconda3/envs/adi_test/lib/python3.8/random.py", line 363, in sample
    raise ValueError("Sample larger than population or is negative")
ValueError: Sample larger than population or is negative
                                                        """