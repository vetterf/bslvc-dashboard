

import platform

pf = platform.system()

if pf == 'Windows':
# sql-server (source db)
  sqlserver_db_config = [
    {
      'Trusted_Connection': 'yes',
    'TrustServerCertificate': 'yes',
      'driver': '{ODBC Driver 18 for SQL Server}',
      'server': r'localhost\SQLSERVERDEV',
      'database': 'BSLVC',
      'UID': 'bslvc_reader',
      'PWD': 'EhiWLj924NvCH7VRmtmp',
      'autocommit': True,
    }
  ]
else:
  sqlserver_db_config = [
    {
      'driver': '/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.4.so.1.1', # for user with podman on linux
      'SERVER' : '127.0.0.1,8444',
      'database': 'BSLVC',
      'UID': 'bslvc_reader',
      'PWD': 'EhiWLj924NvCH7VRmtmp',
      'Trusted_Connection': 'no', 
      'TrustServerCertificate': 'yes',
    }
  ]


