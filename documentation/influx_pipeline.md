#Documentation for seeting up the deep learning pipeline with InfluxDB as data source

###Uploading new data to GPU
* Start up influx ```./influx_dir/influxd```. This starts the local server.
* Sreate ssh tunnel to access DB ```ssh -L 9090:localhost:8086 bprovan@<gpu_ip>```. The local server is hosted on port 8086, this allows from a personal machine on port 9090.
* Credentials for influxdb (username: insight_adi, password: insight_adi) (local machine bprovanbessell, password: as on machine)
* Start writing data to db
* set up grafana to connect to the GPU DB instance

###Running code on gpu cluster
1. ssh to headnode
2. Start influxdb server ```./influxdb2-2.1.1-linux-amd64/influxd ```.
3. ssh to headnode in a new terminal.
4. Create job on gpunode (node001) with ```salloc -p confirm ...```.
5. Connect to that job: ```srun--jobid=<jobid> --pty bash -i```.
6. Set up ssh tunnel to headnode ```ssh -L 9090:localhost:8086 <head_node_ip>```. Unfortunately logs back into the headnode.
7. Re-log on to the job on node001 ```srun--jobid=<jobid> --pty bash -i```.
8. Check that the tunnel has been set up correctly: ```lsof -i:9090```
9. Activate conda environment ```conda activate adi_test```
10. Run training scripts ```python talos_training_<models_etup>.py```
11. Download models to run graph: ```scp -r <gpu_ip>:/home/bprovan/ADI/saved_models/<model_setup>model.tf ADI/saved_models```


####Connecting to Github
Personal access token: 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