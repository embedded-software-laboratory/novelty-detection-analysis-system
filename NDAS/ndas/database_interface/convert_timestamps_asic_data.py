from datetime import datetime
import mysql.connector
import json
import paramiko

def openSSHConnection():
    sshLoginDataFile = open("D:\\Dokumente\\Studium\\Hiwistellen\\SMITH\\novelty-detection-analysis-system\\NDAS\\ndas\\local_data\\sshSettings.json")
    sshLoginData = json.load(sshLoginDataFile)
    try:
        # Establish ssh connection to the database server
        host = "137.226.78.84"
        port = 22
        username = sshLoginData["username"]
        password = sshLoginData["password"]
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port, username, password)
    
    except paramiko.ssh_exception.NoValidConnectionsError:
        return -3
    except TimeoutError: 
        return -4
    except paramiko.ssh_exception.AuthenticationException:
        return -5
        
    return ssh
    
def loadDatabaseConfiguration():
    databaseConfigurationFile = open("D:\\Dokumente\\Studium\\Hiwistellen\\SMITH\\novelty-detection-analysis-system\\NDAS\\ndas\\local_data\\db_asic_scheme.json")
    return json.load(databaseConfigurationFile)

ssh = openSSHConnection()

databaseConfiguration = loadDatabaseConfiguration()
stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select patientid, time from SMITH_ASIC_SCHEME.asic_data_mimic_test"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password']))

results = stdout.readlines()
results = results[1:]

for row in results:
    row = row.split("\t")
    temp = list(row)
    timeNew = datetime.strptime("2138-07-17 20:49:00", "%Y-%m-%d %H:%M:%S").timestamp()
    print(timeNew)
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "update SMITH_ASIC_SCHEME.asic_data_mimic_test set time = \'{}\' where patientid = {} and time = \'{}\'"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], timeNew, temp[0], temp[1]))