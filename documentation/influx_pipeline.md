#Documentation for seeting up the deep learning pipeline with InfluxDB as data source

###Uploading new data to GPU
* Start up influx (```./influx_dir/influxd```). This starts the local server.
* Sreate ssh tunnel to access DB (ssh -L 9090:localhost:8086 bprovan@143.239.81.3). The local server is hosted on port 8086, this allows from a personal machine on port 9090.
* Credentials for influxdb (username: insight_adi, password: insight_adi)
* Start writing data to db
* set up grafana to connect to the GPU DB instance

###Running code on gpu cluster
1. ssh to headnode
2. Start influxdb server ```./influxdb2-2.1.1-linux-amd64/influxd ```.
3. Create job on gpunode (node001) with ```salloc -p confirm ...```.
4. Connect to that job: ```srun--jobid=123 --pty bash -i```.
5. Set up ssh tunnel to headnode ```ssh -L 9090:localhost:8086 10.0.0.250```. Unfortunately logs back into the headnode.
6. Re-log on to the job on node001 ```srun--jobid=123 --pty bash -i```.
7. Activate conda environment ```conda activate adi_test```
8. Run training scripts ```...```