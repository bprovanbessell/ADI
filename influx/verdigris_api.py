"""
Verdigris has sftp server, try to automate collection of latest files/files from a time range

File names of format: machine.YYYY-MM-HHMMSS.gz
Where machine is one of ["JBE10001123", "JBE10001196", "JBE10001268"]


"""
import datetime
import time
import paramiko

def get_files_from_time_range(machine_ids, start_time, end_time):
    pass

# seems to be every 20 minutes, but occasionally it will be 1 minute late or so
# So get the file machine.year-month-day-HHM*.gz
def get_file(machine_id, time):
    host = "107.21.164.35"
    username = "adi"
    password = get_password()

    dt = datetime.datetime.fromtimestamp(time)
    timestr = dt.strftime('%Y-%m-%d%H%M')[0:-1]

    machines = ["JBE10001123", "JBE10001196", "JBE10001268"]

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(hostname=host, username=username, password=password)

    file_base = machines[machine_id] + "." + timestr

    print(file_base)

    sftp_client = ssh_client.open_sftp()

    # seems to be every 20 minutes, but occasionally it will be 1 minute late or so
    # So get the file machine.year-month-day-HHM*.gz
    t1 = time.time()
    # Unfortunately there isn't a better way to do this with sftp, can't do any searching or listing with sftp
    file_names = sftp_client.listdir()
    t2 = time.time()
    print("listdir time: ", t2 - t1)

    # Files are not sorted, so binary search not possible
    # Not sure how to speed this up, as unfortunately we are not
    # guaranteed that the file will be exactly on the 20 minute mark
    # Only seems to be 1 minute off, but I have no idea if that is guaranteed or not...
    file_to_get = ""
    for fn in file_names:
        if file_base in fn:
            file_to_get = fn
            file_result = "verdigris_files/" + file_to_get
            sftp_client.get(remotepath=file_to_get, localpath=file_result)

    if file_to_get == "":
        print("File for that machine at that time does not exist!")

    sftp_client.close()


# Work on more secure way to do this later
def get_password():
    password = ""
    with open("api_files/credentials.txt", 'r') as cred_file:
        for line in cred_file:
            password = line.strip("\n")

    return password


def download_all_with(includes="2022"):
    host = "107.21.164.35"
    username = "adi"
    password = get_password()

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(hostname=host, username=username, password=password)

    sftp_client = ssh_client.open_sftp()

    file_names = sftp_client.listdir()
    print("listdir")

    for fn in file_names:
        if includes in fn:
            file_result = "verdigris_files/2022_files/" + fn
            sftp_client.get(remotepath=fn, localpath=file_result)

    sftp_client.close()


if __name__ == "__main__":

    print(get_password())

    time_test = time.mktime(time.strptime("21.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

    # get_file(0, time_test)
    download_all_with()
