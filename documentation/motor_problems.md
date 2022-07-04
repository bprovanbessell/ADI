25th november general cooling loop (shutdown)
18-20 october general cooling loop - Train on September, test for October
9th february grundfoss


Compare 4 cvae models (all, flux, vib, current) - start with the problem on the 18-20th october
Compare then with the PCA
figure out
- Training data - use whatever there was before as a baseline??
- Run talos training for each of the channels (Vibration, flux, current) (Should have the all measurements one saved at least) (Could possibly use just these models as well for the november error)
- Train a PCA model as well for these errors
- Compare the results of all models between each other - we ideally want to see that certain measurements will
perform better than others, eg be more sensitive to the anomaly. Can we show that the combination is a useful addition?
That combining signals will improve the ability to detect anomalies.

We also will want to compare models to PCA with normal test samples that have no error (do we??)

9th of february has limited measurements (only flux and vibration, no current) - 
So, we can compare 2 models here, but again they should be already trained


tsne, reconstruction error, latent space with and without training set

train, test, anomaly, after
convert points after anomaly to green

train the final all measurements model

test on long consecutive period (with no problems) to make sure error does not change drastically
 - Could do sep, oct, november for Grundfoss/PU7001 (there are unfortunate, so no straight periods > 3 months of undisturbed data)

inference time
- how long does it take to classify a point of data (do for 1000 points, see the total time, divide) 
without loading data/model, just inference time - time to infer a single point (avg) 0.15915201322373668
- how long it takes to train a model - after hyperparameterisation (on one month of data) - 48 minutes?