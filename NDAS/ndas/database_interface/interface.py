from collections import namedtuple
import sys
import csv
import time
import mysql.connector
import json
from datetime import datetime
import os
import paramiko
    
    
def selectBestPatients(db_name, numberOfPatients):
    """
    Returns the patient ids and numbers of records of the patients which have the most records in the given table
    
    Parameters:
        db_name (str) - name of the source database (e.g. "mimic" or "sepsis". There has to be a table named "asic_data_<db_name> in the SMITH_ASIC_SCHEME)
        numberOfPatients (int) - number of patient ids which should be returned
        
    Returns:
        If there occurs some type of an error, an error code is returned which indicates the type of the error:
            -3 - paramiko.ssh_exception.NoValidConnectionsError (means most likely that there is no connection to the i11-VPN)
            -4 - TimeoutError (the ssh connection attempt failed due to a timeout)
            -5 - paramiko.ssh_exception.AuthenticationException (means that there the ssh authentication data are invalid)
            -6 - the database authentication data are invalid
            
        If everything went well, it returns a list of strings which consists of a patient id and the number of records which are stored in the table to the corresponding patient. 
        Each string looks as follows: '<patientid>\t<numberOfRecords>\n', e.g. '898401\t44108\n'.
    """
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh #something went wrong, return the error code
    databaseConfiguration = loadDatabaseConfiguration()
    
    # search for the patient who has the most entries in the given table
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select patientid, entriesTotal from SMITH_ASIC_SCHEME.asic_lookup_{} order by entriesTotal desc limit {};"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], db_name, numberOfPatients))
    errors = stderr.readlines()
    results = stdout.readlines()
    ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): #wrong database authentication data used
        return -6
    results = results[1:] #remove the first entry which consists of the column titles
    return results

def selectBestPatientsWithParameters(db_name, numberOfPatients, parameters):
    """
    Returns the patient ids and numbers of entries per parameter which have the most entries for the specified parameters in the given table
    
    Parameters:
        db_name (str) - name of the source database (e.g. "mimic" or "sepsis". There has to be a table named "asic_data_<db_name> in the SMITH_ASIC_SCHEME)
        numberOfPatients (int) - number of patient ids which should be returned
        parameters (str) - the parameters for which you are searching the patient with the most entries, separated by comma (e.g. "pao2, map, herzfrequenz"). 
        
    Returns:
        If there occurs some type of an error, an error code is returned which indicates the type of the error:
            -3 - paramiko.ssh_exception.NoValidConnectionsError (means most likely that there is no connection to the i11-VPN)
            -4 - TimeoutError (the ssh connection attempt failed due to a timeout)
            -5 - paramiko.ssh_exception.AuthenticationException (means that there the ssh authentication data are invalid)
            -6 - the database authentication data are invalid
            
        If everything went well, it returns a list of strings which consists of a patient id and the number of entries per parameter which are stored in the table to the corresponding patient. 
        The first entry consists of the column titles, formatted like "patientid\t<param1>\t,<param2>\t...<param_n>\n"
        The other entries are strings formatted as follows: "<patientid>\t<number_of_entries_param1>\t>...<number_of_entries_param_n>\n"
    """
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh #something went wrong, return the error code
    databaseConfiguration = loadDatabaseConfiguration()
    # search for the patient who has the most entries in the given table for the specified parameters
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select patientid, {} from (select *, ({}) as numberOfEntries from SMITH_ASIC_SCHEME.asic_lookup_{} order by numberOfEntries desc limit {}) as sub;"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], parameters, parameters.replace(",","+"), db_name, numberOfPatients))
    errors = stderr.readlines()
    results = stdout.readlines()
    ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): #wrong database authentication data used
        return -6 
    return results    
    
def loadPatientData(tableName, patientId):
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh
    databaseConfiguration = loadDatabaseConfiguration()
    
    # selects the patient from the given table with the specified patient id 
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "show columns from SMITH_ASIC_SCHEME.{}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], tableName))
    columnNames = stdout.readlines()
    columnNames = columnNames[2:]
    firstLine = []
    units = ["", "mmHg", "mmHg", "%", "", "°C", "mmHg", "cmH2O", "/min", "/min", "mmol/L", "mmol/L", "µmol/L", "U/L", "mL/cmH2O", "mmHg", "%", "mmol/L", "", "µmol/L", "mmol/L", "10^3/µL", "ng/mL", "mmHg", "mmHg", "%", "mmHg", "", "s", "mmHg", "mL/kg", "U/L", "mmHg", "mmHg", "L/min/m2", "µmol/L", "L/min", "pmol/L", "dyn.s/cm-5/m2", "mmHg", "ng/mL", "dyn.s/cm-5/m2", "cmH2O", "mmHg", "%", "nmol/L", "L/min", "L/min/m2", "ml/m2", "/min", "L/min", "%", "µg/kg/min", "mg/h", "mL/h", "mg/h", "µg/kg/min", "IE/min", "µg/kg/min", "µg/kg/min", "mg/h", "µg/h", "mg", "mg", "mg/h", "mg/h", "mg/h", "µg/kg/min", "mg", "mg", "mg", "mg/h", "µg", "µg/kg/h", "mg", "%", "µg/L", "10^3/µL", "mL", "U/L", "mmol/L", "U/L", "U/L", "ppm", "cmH2O", "", "mL/m2", "mL/Tag", "/min", "%", "", "cmH2O", "mL/kg", "cmH2O", "cmH2O", "cmH2O"]
    index = 0
    for name in columnNames:
        print(name)
        name = name.split()
        if index < len(units):
            firstLine.append(name[0] + "(" + units[index] + ")")
        else:
            firstLine.append(name[0])
        index+=1
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select * from SMITH_ASIC_SCHEME.{} where patientid = {}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], tableName, patientId))
    errors = stderr.readlines()
    result = stdout.readlines()
    #ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']):
        return -6
    result = result[1:]
    if result == []:
        return -1
    convertedRowsTemp = []
    smallestTimestamp = -1
    for row in result:
        row = row.split("\t")
        row = row[1:]
        temp = list(row)
        temp[0] = datetime.strptime(temp[0], "%Y-%m-%d %H:%M:%S").timestamp()
        if temp[0] < smallestTimestamp or smallestTimestamp == -1:
            smallestTimestamp = temp[0]
        row = tuple(temp)
        convertedRowsTemp.append(row)
        convertedRows = [] 
    for row in convertedRowsTemp:
        temp = list(row)
        temp[0] = temp[0] - smallestTimestamp
        convertedRows.append(tuple(temp))
    if not os.path.exists(os.getcwd() + "\\ndas\\local_data\\imported_patients"):
        os.makedirs(os.getcwd() + "\\ndas\\local_data\\imported_patients")
    filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(tableName, patientId)
    file = open(filename, 'w')
    writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_ALL)
    writer.writerow(firstLine)
    for line in convertedRows:
        newLine = list(line)
        writer.writerow(newLine)
    
def loadPatientIds(table):
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh
    databaseConfiguration = loadDatabaseConfiguration()
    
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select distinct patientid from SMITH_ASIC_SCHEME.{};"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], table))
    errors = stderr.readlines()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']):
        ssh.close()
        return -6
    results = stdout.readlines()
    ssh.close()
    return results[1:]
    
def openSSHConnection():
    sshLoginDataFile = open(os.getcwd()+"\\ndas\\local_data\\sshSettings.json")
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
    databaseConfigurationFile = open(os.getcwd()+"\\ndas\\local_data\\db_asic_scheme.json")
    return json.load(databaseConfigurationFile)