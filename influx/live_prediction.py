"""
Should pull data from the machines, and then generate reconstruction error immediately with a model

So, for a specific machine, and specific timeframe
1. Pull data from the otosense api for that timeframe, pull data from verdigris sftp server for that timeframe.
2. Parse that data into a numpy data set.
3. Run model on data set, compute reconstruction error.

"""