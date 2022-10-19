@echo off

echo Check if a compatible python version is installed
echo Checking for python version 3.8...
py -3.8 -c "" >nul 2>nul
if errorlevel 1 (
	echo Version 3.8 not found, checking for python version 3.7...
	py -3.7 -c "" >nul 2>nul
	if errorlevel 1 (
		echo Version 3.7 not found, checking for python version 3.6...
		py -3.6 -c "" >nul 2>nul
		if errorlevel 1 (
			echo You do not have a compatible python version installed, please install python version 3.8, 3.7 or 3.6
			goto :eof
		) else (
			echo Python version 3.6 found!
		)
	) else (
		echo Python version 3.7 found!
	)
) else (
	echo Python version 3.8 found!
)

python -m pip install --upgrade pip > nul
echo Checking for the dependencies...

python -m pip show PyQt5 >NUL 2>nul
if errorlevel 1 (
	echo Installing PyQt5...
	python -m pip install PyQt5
	if errorlevel 1 (
		echo Installation of PyQt5 failed
		goto :eof
	)
) 
python -m pip show numpy >NUL 2>nul
if errorlevel 1 (
	echo Installing numpy...
	python -m pip install numpy
	if errorlevel 1 (
		echo Installation of numpy failed
		goto :eof
	) 
)
python -m pip show pyqtgraph>NUL 2>nul
if errorlevel 1 (
	echo Installing pyqtgraph
	python -m pip install pyqtgraph
	if errorlevel 1 (
		echo Installation of pyqtgraph failed
		goto :eof
	) 
)
python -m pip show pyyaml>NUL 2>nul
if errorlevel 1 (
	echo Installing PyYAML
	python -m pip install pyyaml
	if errorlevel 1 (
		echo Installation of pyyaml failed
		goto :eof
	) 
)
python -m pip show bs4>NUL 2>nul
if errorlevel 1 (
	echo Installing bs4
	python -m pip install bs4
	if errorlevel 1 (
		echo Installation of bs4 failed
		goto :eof
	) 
)
python -m pip show hickle>NUL 2>nul
if errorlevel 1 (
	echo Installing hickle
	python -m pip install hickle
	if errorlevel 1 (
		echo Installation of hickle failed
		goto :eof
	) 
)
python -m pip show lxml>NUL 2>nul
if errorlevel 1 (
	echo Installing lxml
	python -m pip install lxml
	if errorlevel 1 (
		echo Installation of lxml failed
		goto :eof
	) 
)
python -m pip show pandas>NUL 2>nul
if errorlevel 1 (
	echo Installing pandas
	python -m pip install pandas
	if errorlevel 1 (
		echo Installation of pandas failed
		goto :eof
	) 
)
python -m pip show wfdb>NUL 2>nul
if errorlevel 1 (
	echo Installing wfdb
	python -m pip install wfdb
	if errorlevel 1 (
		echo Installation of wfdb failed
		goto :eof
	) 
)
python -m pip show qtwidgets>NUL 2>nul
if errorlevel 1 (
	echo Installing qtwidgets
	python -m pip install qtwidgets
	if errorlevel 1 (
		echo Installation of qtwidgets failed
		goto :eof
	) 
)
python -m pip show seaborn>NUL 2>nul
if errorlevel 1 (
	echo Installing seaborn
	python -m pip install seaborn
	if errorlevel 1 (
		echo Installation of seaborn failed
		goto :eof
	) 
)
python -m pip show kneed>NUL 2>nul
if errorlevel 1 (
	echo Installing kneed
	python -m pip install kneed
	if errorlevel 1 (
		echo Installation of kneed failed
		goto :eof
	) 
)
python -m pip show humanfriendly>NUL 2>nul
if errorlevel 1 (
	echo Installing humanfriendly
	python -m pip install humanfriendly
	if errorlevel 1 (
		echo Installation of humanfriendly failed
		goto :eof
	) 
)
python -m pip show mysql-connector>NUL 2>nul
if errorlevel 1 (
	echo Installing mysql-connector
	python -m pip install mysql-connector
	if errorlevel 1 (
		echo Installation of mysql-connector failed
		goto :eof
	) 
)

python -m pip show paramiko>NUL 2>nul
if errorlevel 1 (
	echo Installing paramiko
	python -m pip install paramiko
	if errorlevel 1 (
		echo Installation of paramiko failed
		goto :eof
	) 
)


echo All necessary dependencies are installed! 