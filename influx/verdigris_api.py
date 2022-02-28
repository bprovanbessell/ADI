"""
Verdigris has sftp server, try to automate collection of latest files/files from a time range

File names of format: machine.YYYY-MM-HHMMSS.gz
Where machine is one of ["JBE10001123", "JBE10001196", "JBE10001268"]


"""

import paramiko

# test to get the file JBE10001196.2022-01-172240.gz

def get_file():
    host = "107.21.164.35"
    username = "adi"
    password = get_password()

    machines = ["JBE10001123", "JBE10001196", "JBE10001268"]

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(hostname=host, username=username, password=password)

    # stdin, stdout, stderr = ssh_client.exec_command("ls")
    # print(stdout.readlines())

    sftp_client = ssh_client.open_sftp()

    file_to_get = "JBE10001196.2022-01-172240.gz"
    file_result = "verdigris_files/" + file_to_get

    sftp_client.get(remotepath=file_to_get, localpath=file_result)
    sftp_client.close()


# Work on more secure way to do this later
def get_password():
    password = ""
    with open("documentation/credentials.txt", 'r') as cred_file:
        for line in cred_file:
            password = line.strip("\n")

    return password


if __name__ == "__main__":

    print(get_password())

    get_file()