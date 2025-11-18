## DEPRECATED
# Support for sqlserver and mysql will be dropped in future versions.

import platform

pf = platform.system()

if pf == 'Windows':
# 
  sqlserver_db_config = [
    {
      'Trusted_Connection': 'yes', # for use with windows and SQLServer
    'TrustServerCertificate': 'yes',
      'driver': '{ODBC Driver 18 for SQL Server}',
      'server': r'localhost\SQLSERVERDEV',
      'database': 'BSLVC',
      'UID': '',
      'PWD': '',
      'autocommit': True,
    }
  ]
else:
  sqlserver_db_config = [
    {
      'driver': '/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.4.so.1.1', # for use with podman on linux
      'SERVER' : '127.0.0.1,8444',
      'database': 'BSLVC',
      'UID': '',
      'PWD': '',
      'Trusted_Connection': 'no', 
      'TrustServerCertificate': 'yes',
    }
  ]


