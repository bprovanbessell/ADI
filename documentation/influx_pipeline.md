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
4. Set up server url in dataset config as ```http://<head_node_ip:port```
5. Set up experiment in python code, and then run ```sbatch -N1 -n1 --partition confirm --job-name=whatever --output=whateer-out.txt run_training.sh```
6. Download models to run graph: ```scp -r <gpu_ip>:/home/bprovan/ADI/saved_models/<model_setup>model.tf ADI/saved_models```


#### Too many open files
```ulimit -n 10000```
