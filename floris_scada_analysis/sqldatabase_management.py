# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import pandas as pd
from datetime import datetime


def generate_models_py(table_names, table_upsampling, fn_channel_defs,
                       wide_format, turbine_range, repository_name,
                       settingsfile_name, fout):
    """A function to automatically generate settings.py for an SQL
    wind farm data repository.

    Args:
        table_names ([list]): List of SQL table names
        table_upsampling ([list]): List of bools defining whether
        table will contain upsampled data or not. If yes, then
        additional table entries will be made to include _mean,
        _median, _std, _min and _max for all numerical variables.
        fn_channel_defs ([str]): Path to xlsx with channel definitions.
        wide_format ([bool]): Definition whether table is WIDE or TALL.
        turbine_range ([list]): List of turbine names.
        repository_name ([str]): Name of Python repository
        settingsfile_name ([str]): Name of settings file for Python/SQL
        fout ([str]): Path defining desired output directory for models.py

    Raises:
        Exception: Will raise an error when models.py already exists,
        and when additionally models.py.old is already taken.
    """

    # Initialize default template
    modelspy_str = \
    '''
    # AUTOMATICALLY GENERATED FUNCTION BY 'GENERATE_MODELSPY.PY'
    #      DATE: ''' + datetime.now().strftime('%H:%M:%S on %A, %B the %dth, %Y') + ''''

    from sqlalchemy import create_engine
    from sqlalchemy import Table, Column, String, Integer, DateTime, UniqueConstraint, REAL
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.engine.url import URL

    from ''' + repository_name + ''' import ''' + settingsfile_name + '''

    DeclarativeBase = declarative_base()


    def db_connect():
        """
        Performs database connection using database settings from settings.py.
        Returns sqlalchemy engine instance
        """
        print(''' + settingsfile_name + '''.DATABASE)
        return create_engine(URL(**''' + settingsfile_name + '''.DATABASE), pool_timeout=60*60*24)


    def create_tables(engine):
        """ Initialize all tables """
        DeclarativeBase.metadata.create_all(engine)


    def recreate_all(engine):
        """ Delete all tables and data and recreate base tables """
        print("DROPPING AND RECREATING ALL TABLES")
        DeclarativeBase.metadata.drop_all(engine)
        DeclarativeBase.metadata.create_all(engine)


    def recreate_by_type(engine, table_name):
        """ Delete tables by type and recreate"""

        table_name = table_name.lower()
    '''

    # Populate recreate_by_type(...) function
    for table_name in table_names:
        modelspy_str = modelspy_str + '''
        if table_name == "''' + table_name + '''":
            DeclarativeBase.metadata.drop_all(engine, tables=[''' + table_name + '''_filenames_class.__table__, ''' + table_name + '''_class.__table__])
            DeclarativeBase.metadata.create_all(engine, tables=[''' + table_name + '''_filenames_class.__table__, ''' + table_name + '''_class.__table__])
        '''


    # Add individual classes for table_name_filenames and table_names
    for iii in range(len(table_names)):
        table_name = table_names[iii]

        modelspy_str = modelspy_str + '''
    class ''' + table_name + '''_filenames_class(DeclarativeBase):
        """Sqlalchemy filenames model"""
        __tablename__ = "''' + table_name + '''_filenames"
        id = Column(Integer, primary_key=True, autoincrement=True)
        filename = Column('filename', String, nullable=False, primary_key=True, unique=True)

    class ''' + table_name + '''_class(DeclarativeBase):
        """Sqlalchemy data model"""
        __tablename__ = "''' + table_name + '''"

        index = Column('index', Integer, primary_key=True, autoincrement=True)'''

        # Add individual variable definitions
        df_defs = pd.read_excel(fn_channel_defs, sheet_name='scada_data')

        # Extract channel mappings and determine to-be-removed channels
        original_name = df_defs[df_defs.columns[0]]  # Raw dataset names
        database_name = df_defs[df_defs.columns[1]]  # Desired database names
        database_omit = df_defs['remove_id']  # To-be-omitted fields
        variable_type = df_defs['var_type']  # Variable type
        remove_fields = [
            database_name[i]
            for i in range(len(original_name))
            if database_omit[i].lower() == "yes"
        ]

        # Create index/unique variables first
        if 'turbid' in database_name.values and not wide_format:
            remove_fields.append('turbid')
            modelspy_str = modelspy_str + ''' 

        # Time and turbine are the indexed columns (OVERWRITE)
        time = Column('time', DateTime, index=True)
        turbid = Column('turbid', Integer,nullable=True, index=True)
        UniqueConstraint('time', 'turbid', name='timeturbid')
        '''
        else:
            modelspy_str = modelspy_str + ''' 

        # Time is the indexed column (OVERWRITE)
        time = Column('time', DateTime, primary_key=True, unique=True, index=True)
        '''

        # Create all necessary db variables in models.py
        remove_fields.append('time')  # Do not repeat time variable
        if wide_format:
            remove_fields.append('turbid')
            db_name_new = []
            vb_type_new = []
            for di in range(len(database_name)):
                if database_name[di] not in remove_fields:
                    for ti in turbine_range:
                        if table_upsampling[iii]:
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_mean")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_median")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_std")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_min")
                            vb_type_new.append(variable_type[di])
                            db_name_new.append(database_name[di] + "_" + str(ti) + "_max")
                            vb_type_new.append(variable_type[di])
                        else:
                            db_name_new.append(database_name[di] + "_" + str(ti))
                            vb_type_new.append(variable_type[di])
            database_name = db_name_new
            variable_type = vb_type_new

        for i in range(len(database_name)):
            if database_name[i] not in remove_fields:
                modelspy_str = modelspy_str + '''
        ''' + database_name[i] + ''' = Column("''' + database_name[i] + '''", ''' + variable_type[i] + ''', nullable=True)'''
        modelspy_str = modelspy_str + '''
        '''


    # WRITE TO FILE, but make sure no models.py file gets overwritten
    if os.path.exists(fout):
        if os.path.exists(fout + '.old'):
            raise Exception('Both a models.py and models.py.old file exist in fout. Remove one or both.')
        else:
            os.rename(fout, fout+'.old')
            print("Found existing file. Renamed 'models.py' to 'models.py.old'.")

    # Write a new models.py
    text_file = open(fout, "w")
    n = text_file.write(modelspy_str)
    text_file.close()
    print("Finished writing 'models.py' to " + fout + ".")

    return None
