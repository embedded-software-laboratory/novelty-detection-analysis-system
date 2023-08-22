from collections import namedtuple
import sys
import csv
import time
import mysql.connector
import json
from datetime import datetime
import os
import paramiko
import logging

    
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

    results = stdout.readlines()
    errors = stderr.readlines()

    ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): #wrong database authentication data used
        return -6
    elif len(errors) > 0:
        for error in errors:
            logging.error(error)
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

    results = stdout.readlines()
    errors = stderr.readlines()

    ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): #wrong database authentication data used
        return -6
    elif len(errors) > 0:
        for error in errors:
            logging.error(error)
    return results    


def loadPatientData(tableName, patientId):
    """
    Loads all data of the specified patient from the given table and stores it into a csv table.
    
    This method is called by the NDAS when the user loads a patient using the "load patient from database" functionality. The data are stored into a csv table 
    to avoid the loading time for further accesses. (The NDAS perfomrs a check if there already exists this csv table before calling this function). 

    Parameters:
        tableName (str) - the name of the table / database which contains the data. There must either exist a table with this name in the scheme ASIC_DATA_SCHEME or you'll have to implement the compatibility for other darabases by hand. 
        patientId (int) - the id of the patient
        
        Returns:
        If there occurs some type of an error, an error code is returned which indicates the type of the error:
            -1 - the specified patient does not exist
            -3 - paramiko.ssh_exception.NoValidConnectionsError (means most likely that there is no connection to the i11-VPN)
            -4 - TimeoutError (the ssh connection attempt failed due to a timeout)
            -5 - paramiko.ssh_exception.AuthenticationException (means that there the ssh authentication data are invalid)
            -6 - the database authentication data are invalid 
        If everything went well, nothing is returned. 
    """
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh
    databaseConfiguration = loadDatabaseConfiguration()
    firstLine = [None]
    convertedRows = []

    if tableName == "smith_omop": # load patient from the smith_omop database
        original_timestamps = "("
        stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_OMOP -e "with a as (select distinct drug_exposure_start_datetime as pTime from SMITH_OMOP.drug_exposure where person_id = {}), b as (select distinct measurement_datetime as pTime from SMITH_OMOP.measurement where person_id = {}), c as (select distinct observation_datetime as pTime from SMITH_OMOP.observation where person_id = {}), d as (select distinct procedure_datetime as pTime from SMITH_OMOP.procedure_occurrence where person_id = {}) select pTime from (select * from a union select * from b union select * from c union select * from d) as res order by pTime;"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], patientId, patientId, patientId, patientId)) # get all timestamps at which data exist in the database to the patient
        timestamps = stdout.readlines()[1:]
        for timestamp in timestamps:
            original_timestamps = original_timestamps + " '" + timestamp + "', "
            convertedRows.append([(datetime.strptime(timestamp.replace("\n", ""), '%Y-%m-%d %H:%M:%S') - datetime.strptime(timestamps[0].replace("\n", ""), '%Y-%m-%d %H:%M:%S')).total_seconds()])
        original_timestamps = original_timestamps[:-2] # remove the last comma
        original_timestamps = original_timestamps + ")"
       
        conceptColumns = {"drug_exposure": "drug_concept_id", "measurement": "measurement_concept_id", "observation": "observation_concept_id"}
        timeColumns = {"drug_exposure": "drug_datetime", "measurement": "measurement_datetime", "observation": "observation_datetime"}
        parameters = {"3038018": "24h-Bilanz", "3043148": "Atemfrequenz", "46235169": "Amylase", "3003396":  "BE_arteriell", "3008152": "Bicarbonat_arteriell", 
                      "40757494": "Bilirubin_ges", "3007220": "CK", "21490582": "Compliance", "3012888": "DAP", "3024882": "FiO2", "3013721": "GOT", "3005755": "GPT", 
                      "3027018": "Herzfrequenz", "43534077": "Harnstoff", "3029943": "Horowitz-Quotient (ohne_Temp_Korrektur)", "3009542": "Hämatokrit", 
                      "40762351": "Hämoglobin", "21490724": "I:E", "3032080": "INR", "40762887": "Kreatinin", "3025926": "Körperkerntemperatur", 
                      "3018405": "Laktat (arteriell)", "3010813": "Leukozyten", "3004905": "Lipase", "3027598": "MAP", "21490855": "PEEP", "3004249": "SAP", 
                      "3016502": "SaO2", "40762500": "SpO2", "3005797": "SzvO2", "21490675": "ZVD", "3019977": "pH (arteriell)", "3013466": "pTT", 
                      "3027946": "paCO2_(ohne_Temp-Korrektur)", "3027801": "paO2_(ohne_Temp-Korrektur)", "40479778": "Extrakorporaler Blutfluss", 
                      "3661655": "EVLWI", "3029790": "CK-MB", "21490776": "PCWP", "3028074": "MPAP", "21490712": "Herzindex (kontinuierlich)", "3048529": "Troponin", 
                      "3007461": "Thrombozyten", "40757478": "Albumin", "3036518": "Schlagvolumen (kontinuierlich)", "3005555": "Herzzeitvolumen (kontinuierlich)", 
                      "3052295": "BNP", "21490880": "Schlagvolumenindex (kontinuierlich)", "3009713": "PVRI", "3017188": "DPAP", "3046279": "PCT", "3008047": "SVRI", 
                      "4254061": "GEDVI", "3012810": "P_EI", "3005606": "SPAP", "3035357": "etCO2", "46234770": "CRP", "3017594": "Tidalvolumen", "4353937": "FeO2", 
                      "42527138": "deltaP", "3002309": "Herzzeitvolumen (Bolus)", "3000569": "Herzindex (Bolus)", "3008952": "Schlagvolumen (Bolus)", 
                      "3024530": "Schlagvolumenindex (Bolus)", "3026892": "Atemfrequenz (spontan)", "4089495": "Extrakorporaler_Gasfluss_(O2)", 
                      "1321341": "Norepinephrin", "753626": "Propofol", "1343916": "Epinephrin", "35201749": "Vasopressin", "1719036": "Dobutamin", 
                      "40173184": "Levosimendan", "956874": "Furosemid", "19078219": "Sufentanil", "19003953": "Rocuronium", "975125": "Hydrocortison", 
                      "46287617": "Midazolam", "1110410": "Morphin", "785649": "Ketanest", "42628988": "Milrinon", "1550557": "Prednisolon", "1518254": "Dexamethason", 
                      "35603427": "Fentanyl", "35603036": "Dexmedetomidin"} # maps the concept ids to the according parameter name

        for table, concept_id in conceptColumns.items():
            if table == "drug_exposure":
                value_column = "quantity"
            else:
                value_column = "value_as_number"
            stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_OMOP -e "select distinct {} from SMITH_OMOP.{} where person_id = {}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], concept_id, table, patientId)) #iterate over all tables that may contain data to the patient and get the concept ids to which data exist
            concept_ids = (stdout.readlines()[1:]),
            concept_ids = concept_ids[0]
            if concept_ids != []:
                for loaded_concept_id in concept_ids:
                    firstLine.append(parameters[loaded_concept_id.replace("\n", "")])                                                                                                                                                                                                                                                                                                                                                                                                                                        
                    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_OMOP -e "select {}, {} from SMITH_OMOP.{} where person_id = {} and {} in {} and {} = {} order by {} asc"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], value_column, timeColumns[table], table, patientId, timeColumns[table], original_timestamps, conceptColumns[table], loaded_concept_id, timeColumns[table])) #select the data that exists in the table at the given time
                    results = stdout.readlines()[1:] 
                    counter = 0;
                    for result in results:  
                        resultList = result.split("\t")                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                        while timestamps[counter] != resultList[1]: # skip all timestamps to which not exist data with this concept id 
                            convertedRows[counter].append("NULL")
                            counter+=1 #since the results come in sorted order, we do not need to iterate over all timestamps in every loop
                            if counter >= len(timestamps): 
                                break
                        if counter >= len(timestamps):
                            break
                        convertedRows[counter].append(resultList[0])
                        counter+=1
                    while counter != len(timestamps):
                        convertedRows[counter].append("NULL")
                        counter+=1

                        

    else: # if nothing else applies, load data from the smith_asic_scheme
        # first, get all column names of the table and add the respective unit to the column titel (this is done so that the NDAS can show the unit for every parameter). 
        stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "show columns from SMITH_ASIC_SCHEME.{}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], tableName)) #get all column names
        columnNames = stdout.readlines()
        columnNames = columnNames[2:] #remove the first two elements of the result list (the first element contains the column names of the query result and the second element (the first query result element) is the "patientid" column, which is not further needed).
        units = ["mmHg", "mmHg", "%", "", "°C", "mmHg", "cmH2O", "/min", "/min", "mmol/L", "mmol/L", "µmol/L", "U/L", "mL/cmH2O", "mmHg", "%", "mmol/L", "", "µmol/L", "mmol/L", "10^3/µL", "ng/mL", "mmHg", "mmHg", "%", "mmHg", "", "s", "mmHg", "mL/kg", "U/L", "mmHg", "mmHg", "L/min/m2", "µmol/L", "L/min", "pmol/L", "dyn.s/cm-5/m2", "mmHg", "ng/mL", "dyn.s/cm-5/m2", "cmH2O", "mmHg", "%", "nmol/L", "L/min", "L/min/m2", "ml/m2", "/min", "L/min", "%", "µg/kg/min", "mg/h", "mL/h", "mg/h", "µg/kg/min", "IE/min", "µg/kg/min", "µg/kg/min", "mg/h", "µg/h", "mg", "mg", "mg/h", "mg/h", "mg/h", "µg/kg/min", "mg", "mg", "mg", "mg/h", "µg", "µg/kg/h", "mg", "%", "µg/L", "10^3/µL", "mL", "U/L", "mmol/L", "U/L", "U/L", "ppm", "cmH2O", "", "mL/m2", "mL/Tag", "/min", "%", "", "cmH2O", "mL/kg", "cmH2O", "cmH2O", "cmH2O"] #the units for all colums. TODO: This has to be solved in another way since at this point, the table columns need to be in the correct order, and this cannot be guaranteed for upcoming tables in the SMITH_ASIC_SCHEME. 
        index = 0
        columnNames = columnNames[1:]
        for name in columnNames:
            name = name.split()
            if index < len(units):
                firstLine.append(name[0] + "(" + units[index] + ")") # add the unit to the column title
            else:
                firstLine.append(name[0]) 
            index+=1
            
        # load the actual data for the specified patient    
        stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select * from SMITH_ASIC_SCHEME.{} where patientid = {}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], tableName, patientId))
        result = stdout.readlines()
        errors = stderr.readlines()
        ssh.close()
        if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): # wrong database authentication data
            return -6
        elif len(errors) > 0:
            for error in errors:
                logging.error(error)
        result = result[1:]
        if result == []:
            return -1
            
        convertedRowsTemp = []
        smallestTimestamp = -1
        for row in result:
            row = row.split("\t")
            temp = list(row)
            temp.pop(0)
            temp[0] = float(temp[0])
            if temp[0] == 17584.0000000000:
                print(row)
            if temp[0] < smallestTimestamp or smallestTimestamp == -1:
                smallestTimestamp = temp[0]
            row = tuple(temp)
            convertedRowsTemp.append(row)
            convertedRows = [] 
        for row in convertedRowsTemp:
            temp = list(row)
            temp[0] = temp[0] - smallestTimestamp
            temp[-1] = temp[-1].replace("\n", "")
            convertedRows.append(tuple(temp))
    if not os.path.exists(os.getcwd() + "\\ndas\\local_data\\imported_patients"):
        os.makedirs(os.getcwd() + "\\ndas\\local_data\\imported_patients") #create the folder where the csv table has to be stored
    #write the data to the csv file
    filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(tableName, patientId)
    file = open(filename, 'w', newline="")
    writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_ALL)
    writer.writerow(firstLine)
    for line in convertedRows:
        newLine = list(line)
        print(newLine)
        writer.writerow(newLine)

def loadPatientIds(table, omop_data_source=None):
    """
    Returns all patient ids in a list which occur in the specified table / database.
    
    Parameters:
        table (str) - the name of the table. There must exist a table with this nam ein the ASIC_DATA_SCHEME. For other databases, the functionality has to be extended manually. 
        omop_data_source (str) - the name of the data source, from which the patient ids should be loaded. Only necessary for the omop_database (can be ommited in other cases). (The omop database contains data from other databases like MIMIC3 or HIRID, which were converted into the OMOP CDM)
        
    Returns:
    If there occurs some type of an error, an error code is returned which indicates the type of the error:
        -3 - paramiko.ssh_exception.NoValidConnectionsError (means most likely that there is no connection to the i11-VPN)
        -4 - TimeoutError (the ssh connection attempt failed due to a timeout)
        -5 - paramiko.ssh_exception.AuthenticationException (means that there the ssh authentication data are invalid)
        -6 - the database authentication data are invalid     
        
    If everything went well, a list of integers is returned
    """
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh
    databaseConfiguration = loadDatabaseConfiguration()
    
    #retrieve the data from the database
    if table == "smith_omop": # load patient ids from the omop database
        if omop_data_source == "MIMIC3":
            stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_OMOP -e "select distinct person_id from SMITH_OMOP.person where person_source_value like \'MIMIC%\' and person_source_value not like \'MIMICIV%\';"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password']))
        else: 
            stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_OMOP -e "select distinct person_id from SMITH_OMOP.person where person_source_value like \'{}%\';"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], omop_data_source))
    else: #assume that the given table name is a table in the SMITH_ASIC_SCHEME database
        stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select distinct patientid from SMITH_ASIC_SCHEME.{};"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], table))
    results = stdout.readlines()
    errors = stderr.readlines()
    ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): #database login data are wrong
        ssh.close()
        return -6
    elif len(errors) > 0:
        for error in errors:
            logging.error(error)

    return results[1:]


def openSSHConnection():
    """
    Establishes a ssh connection to the i11 smith-interface server which is needed in order to access the SMITH databases.
    The authentication data for the server must be stored in the file "sshSettings.json" in the folder local_data. (This should be done by the user using the NDAS). 
    
    Returns:
    If there occurs some type of an error, an error code is returned which indicates the type of the error:
        -3 - paramiko.ssh_exception.NoValidConnectionsError (means most likely that there is no connection to the i11-VPN)
        -4 - TimeoutError (the ssh connection attempt failed due to a timeout)
        -5 - paramiko.ssh_exception.AuthenticationException (means that there the ssh authentication data are invalid)
        
    If the connection could successfully be established, it returns a paramiko-SSH-client
    """
    sshLoginDataFile = open(os.getcwd()+"\\ndas\\local_data\\sshSettings.json")
    sshLoginData = json.load(sshLoginDataFile)
    try:
        # Establish ssh connection to the interface server
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
    """
    Returns a json object which contains the login data for the smith database.
    The authentication data must be stored in the file "db_asic_scheme.json" in the folder local_data. (This should be done by the user using the NDAS). 
    """
    databaseConfigurationFile = open(os.getcwd()+"\\ndas\\local_data\\db_asic_scheme.json")
    return json.load(databaseConfigurationFile)
    
def testDatabaseConnection():
    """
    Test if a connection to the database can successfully be established
    
    Returns
        1 - Connection could successfully be established
        -3 - paramiko.ssh_exception.NoValidConnectionsError (means most likely that there is no connection to the i11-VPN)
        -4 - TimeoutError (the ssh connection attempt failed due to a timeout)
        -5 - paramiko.ssh_exception.AuthenticationException (means that there the ssh authentication data are invalid)
        -6 - the database authentication data are invalid 
    """
    ssh = openSSHConnection()
    if ssh in [-3,-4,-5]:
        return ssh
    databaseConfiguration = loadDatabaseConfiguration()
    stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select patientid from SMITH_ASIC_SCHEME.asic_data_sepsis limit 1;"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password']))
    results = stdout.readlines()
    errors = stderr.readlines()
    ssh.close()
    if len(errors) > 0 and errors[0] == "ERROR 1045 (28000): Access denied for user '{}'@'interface.smith.embedded.rwth-aachen.de' (using password: YES)\n".format(databaseConfiguration['username']): #database login data are wrong
        ssh.close()
        return -6
        
    else:
        return 1