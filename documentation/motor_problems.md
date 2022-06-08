25th november general cooling loop (shutdown)
18-20 october general cooling loop
9th february grundfoss


Compare 4 cvae models (all, flux, vib, current) - start with the problem on the 18-20th october
Compare then with the PCA
figure out
- Training data - use whatever there was before as a baseline??
- Run talos training for each of the channels (Vibration, flux, current) (Should have the all measurements one saved at least) (Could possibly use just these models as well for the november error)
- Train a PCA model aswell for these errors

We also will want to compare models to PCA with normal test samples that have no error, so we can see overfitting etc
I'll have to save multiple models as well, as probably the later models will have more overfitting (ones with lower test error)