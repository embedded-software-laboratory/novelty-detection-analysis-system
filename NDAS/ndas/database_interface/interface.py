from collections import namedtuple
import sys
import csv
import time
import mysql.connector
import json
from datetime import datetime
import os
import paramiko

def startInterface(argv):
	Table = namedtuple('Table', ['name', 'type', 'relevant_columns', 'parameter_identifier_column'])
	Parameter = namedtuple('Parameter', ['name','database', 'tables', 'parameter_identifier'])
	currentPath = os.path.dirname(__file__)

	sshLoginDataFile = open(os.getcwd()+"\\ndas\\local_data\\sshSettings.json")
	sshLoginData = json.load(sshLoginDataFile)

	databaseConfigurationFile = open(os.getcwd()+"\\ndas\\local_data\\db_asic_scheme.json")
	databaseConfiguration = json.load(databaseConfigurationFile)
	# Establish ssh connection to the database server
	host = "137.226.78.84"
	port = 22
	username = sshLoginData["username"]
	password = sshLoginData["password"]
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(host, port, username, password)

	#----------------------
	#-----Tables-----------
	#----------------------
	tables = []
	with open(currentPath + "\\Interface_tables.csv") as parameter_csv:
		csv_reader_object = csv.reader(parameter_csv, delimiter=";")
		firstLineFlag = True
		for row in csv_reader_object:
			if firstLineFlag:
				firstLineFlag = False
				continue
			tables.append(Table(name=row[0], type=int(row[1]), relevant_columns=row[2], parameter_identifier_column=row[3]))


	#----------------------
	#----Parameter list----
	#----------------------
	parameters = []
	with open(currentPath + "\\Interface_parameter.csv") as parameter_csv:
		csv_reader_object = csv.reader(parameter_csv, delimiter=";")
		firstLineFlag = True
		for row in csv_reader_object:
			if firstLineFlag:
				firstLineFlag = False
				continue
			parameterTables = row[2].split(",")
			tableReferences = []
			for table in tables:
				for parameterTable in parameterTables:
					if table.name == parameterTable:
						tableReferences.append(table)
			if len(tableReferences) == 0:
				print("Error reading parameter table: Table(s) not found in parameter " + row[0] + " (Database " + row[1] + ")")
				continue
			if tableReferences[0].type == 1:
				parameterIdentifier = row[3].split(",")
				parameterIdentifier = list(map(int, parameterIdentifier))
				parameters.append(Parameter(name=row[0], database=row[1], tables=tableReferences, parameter_identifier=parameterIdentifier))
			elif tableReferences[0].type == 2:
				parameters.append(Parameter(name=row[0], database=row[1], tables=tableReferences, parameter_identifier=row[3]))

	#parameterFile = open(currentPath + "\\" + argv[1])
	#connectionParameters = json.load(parameterFile)
	#parameterFile.close()
	#connection = mysql.connector.connect(host=connectionParameters['host'],
	#                                     database=connectionParameters['dbname'],
	#                                     user=connectionParameters['user'],
#                                     password=connectionParameters['password'])
	if argv[2] == "dataDensity":
		if argv[3] == "patientid":
			if argv[5] == "entriesTotal":
				cur.execute("select count(*) from SMITH_ASIC_SCHEME.asic_data_mimic where patientid = {}".format(argv[4]))
				result = cur.fetchall()
				print(result)
				cur.close()
			else:
				index = 0
				identifier = ""
				for arg in argv:
					if index < 5:
						index+=1
						continue
					found = False
					for parameter in parameters:
						if parameter.name == arg and parameter.database == "asic":
							found = True
							identifier = identifier + parameter.parameter_identifier + ", "
					if found == False:
						print("Unknown parameter " + arg + " in database asic")
					index+=1
				identifier = identifier[:-2]
				identifierComplete = identifier.replace(", ", " and ") + " IS NOT NULL"
				cur.execute("select count(*) from SMITH_ASIC_SCHEME.asic_data_mimic where patientid = {} and {}".format(argv[4], identifierComplete))
				result = cur.fetchall()
				print(result)
				cur.close()
		elif argv[3] == "bestPatients":
			if argv[4] == "entriesTotal":
				sqlFile = open("dataDensity.sql")
				dataDensityProcedure = sqlFile.read().replace("\n", " ")
				sqlFile.close()
				cur.execute(dataDensityProcedure)
				cur.callproc('dataDensity', [argv[5]])
				for result in cur.stored_results():
					print(result.fetchall())
				cur.execute('drop procedure if exists dataDensity; ')
				cur.close()
			else: 
				cur.execute('drop procedure if exists dataDensityWithParameter; ')
				index = 0
				identifier = ""
				for arg in argv:
					if index < 4:
						index+=1
						continue
					if index == len(argv)-1:
						break
					found = False
					for parameter in parameters:
						if parameter.name == arg and parameter.database == "asic":
							found = True
							identifier = identifier + parameter.parameter_identifier + " and "
					if found == False:
						print("Unknown parameter " + arg + " in database asic")
					index+=1
				identifier = identifier[:-4]				
				sqlFile = open("dataDensityWithParameter.sql")
				dataDensityProcedure = sqlFile.read().replace("\n", " ").replace("$placeholder", identifier)
				sqlFile.close()
				cur.execute(dataDensityProcedure)
				cur.callproc('dataDensityWithParameter', [argv[len(argv)-1]])
				for result in cur.stored_results():
					print(result.fetchall())
				cur.execute('drop procedure if exists dataDensityWithParameter; ')
				cur.close()		
	elif argv[2] == "selectPatient":
		stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "show columns from SMITH_ASIC_SCHEME.{}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], argv[4]))
		columnNames = stdout.readlines()
		columnNames = columnNames[2:]
		firstLine = []
		for name in columnNames:
			name = name.split()
			firstLine.append(name[0])
		stdin, stdout, stderr = ssh.exec_command('mysql -h{} -u{} -p{} SMITH_SepsisDB -e "select * from SMITH_ASIC_SCHEME.{} where patientid = {}"'.format(databaseConfiguration['host'], databaseConfiguration['username'], databaseConfiguration['password'], argv[4], argv[3]))
		result = stdout.readlines()
		result = result[1:]
		convertedRows = []
		for row in result:
			row = row.split("\t")
			row = row[1:]
			temp = list(row)
			temp[0] = datetime.strptime(temp[0], "%Y-%m-%d %H:%M:%S").timestamp()
			row = tuple(temp)
			convertedRows.append(row)
		if not os.path.exists(os.getcwd() + "\\ndas\\local_data\\imported_patients"):
			os.makedirs(os.getcwd() + "\\ndas\\local_data\\imported_patients")
		filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(argv[4], argv[3])
		file = open(filename, 'w')
		writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_ALL)
		writer.writerow(firstLine)
		for line in convertedRows:
			newLine = list(line)
			writer.writerow(newLine)
	else:
		found = False
		for parameter in parameters:
			if parameter.name == argv[3] and parameter.database == argv[2]:
				for table in parameter.tables:
					identifier = ""
					result = ""
					firstLine = table.relevant_columns.split(",")
					if table.type == 1:
						for id in parameter.parameter_identifier:
							identifier = identifier + table.parameter_identifier_column + ' = ' + str(id) + " or "
						identifier = identifier[:-4]
						cur.execute('SELECT ' + table.relevant_columns + ' FROM ' + table.name + ' WHERE ' + identifier)
						result = cur.fetchall()
					elif table.type == 2:
						index = 0
						for arg in argv:
							if index < 3:
								index+=1
								continue
							if arg == "noNullValues" and index == len(argv)-1:
								break;
							found2 = False
							for parameter2 in parameters:
								if parameter2.name == arg and parameter2.database == argv[2]:
									found2 = True
									firstLine.append(parameter2.parameter_identifier)
									identifier = identifier + parameter2.parameter_identifier + ", "
							if found2 == False:
								print("Unknown parameter " + arg + " in database " + argv[2])
							index+=1
						identifier = identifier[:-2]
						if argv[len(argv)-1] == "noNullValues":
							identifierComplete = identifier.replace(", ", " and ") + " IS NOT NULL"
						else:
							identifierComplete = identifier.replace(", ", " or ") + " IS NOT NULL"
						cur.execute('SELECT ' + table.relevant_columns + ', ' + identifier + ' FROM ' + table.name + ' WHERE ' + identifierComplete)
						result = cur.fetchall()
					file = open("queryResult.csv", 'a')
					writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_ALL)
					writer.writerow(firstLine)
					for line in result:
						newLine = list(line)
						writer.writerow(newLine)	
				found = True		
				break
		if found == False:
			print("Unknown parameter " + argv[3] + " in database " + argv[2])
	ssh.close()