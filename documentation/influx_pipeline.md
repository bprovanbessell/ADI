#Documentation for seeting up the deep learning pipeline with InfluxDB as data source

###Uploading new data to GPU
* Start up influx (./influx_dir/influxd). This starts the local server.
* Sreate ssh tunnel to access DB (ssh -L 9090:localhost:8086 bprovan@143.239.81.3). The local server is hosted on port 8086, this allows from a personal machine on port 9090.
* Credentials for influxdb (username: insight_adi, password: insight_adi)
* Start writing data to db
* set up grafana to connect to the GPU DB instance