# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   MysqlToES.py
 
@Time    :   2020/10/3 3:41 下午
 
@Desc    :   mysql数据同步到ES
 
"""
import argparse
from configparser import ConfigParser

import requests
import json
import pymysql as mdb
from datetime import datetime
import sys


class MySQLToElasticSearch:

    def __init__(self):
        """ Setting up required variables.
        """
        self.conf_file = ''
        self.__config = self.parse_config(self.conf_file)

        self.cursor = self.__get_db_cursor(self.__config)
        self.__tables = self.__get_table_names()
        self.structure = self.get_db_structure(self.__tables)

        self.es_url = self.__config["es_url"]
        self.es_index = self.__config["es_index"]
        self.session = requests.Session()

    def parse_config(self, conf_file):
        config = ConfigParser()
        config.read(conf_file)
        confs = {}
        confs["db_host"] = config.get("mysql", "host")
        confs["db_port"] = config.getint("mysql", "port")
        confs["db_user"] = config.get("mysql", "username")
        confs["db_pass"] = config.get("mysql", "password")
        confs["db_name"] = config.get("mysql", "database")
        confs["es_url"] = config.get("elasticsearch", "es_url")
        confs["es_index"] = config.get("elasticsearch", "es_index")
        return confs

    def __get_db_cursor(self, config):
        try:
            self.__conn = mdb.connect(host=self.__config["db_host"],
                                      user=self.__config["db_user"],
                                      port=self.__config["db_port"],
                                      passwd=self.__config["db_pass"],
                                      db=self.__config["db_name"])
            self.__conn.autocommit(True)
            return self.__conn.cursor()
        except Exception as e:
            print("[-]Error: While connecting to database, got \n%s." % str(e))
            sys.exit(-1)

    def __get_table_names(self):
        query = "SHOW TABLES"
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return [res[0] for res in result]
        except Exception as e:
            print("[-]Error: While fetching table names, got \n%s." % str(e))
            sys.exit(-1)

    def __get_column_names(self, table_name):
        query = "DESC `%s`" % table_name
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return [res[0] for res in result]
        except Exception as e:
            print("[-]Error: While fetching column names, got \n%s." % str(e))
            sys.exit(-1)

    def get_db_structure(self, tables):
        result = []
        for table in tables:
            res = {}
            res["table_name"] = table
            res["columns"] = self.__get_column_names(table)
            result.append(res)
        return result

    def get_data(self, structure):
        """
        查询数据
        """
        result = []
        query = "SELECT * FROM %s WHERE 1" % structure["table_name"]
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            for row in rows:
                if len(row) == len(structure["columns"]):
                    res = {}
                    index = 0
                    for key in structure["columns"]:
                        res[key] = row[index]
                        index += 1
                    result.append(res)
            return result
        except Exception as e:
            print("[-]Error: While fetching date from tables, got \n%s." % str(e))
            sys.exit(-1)

    def push_to_elasticsearch(self, data, doc_type):
        prepared_url = "%s/%s/%s" % (self.es_url, self.es_index, doc_type)
        try:
            data = json.dumps(data)
        except:
            for key in data.keys():
                if isinstance(data[key], datetime):
                    data[key] = data[key].isoformat()
                data[key] = str(data[key])
            data = json.dumps(data)

        try:
            req = self.session.post(prepared_url, data=data)
            if req.status_code == 201:
                return True
            else:
                print("[+]Error: Unexpected response from ElasticSearch.\
                         Got status_code - %d" % req.status_code)
        except Exception as e:
            print("[-]Error: While dumping to elasticsearch. Got %s" % str(e))
            print(data)
            sys.exit(-1)

    def __del__(self):
        try:
            self.__conn.close()
            self.session.close()
        except:
            pass


def main():
    try:
        my_dump = MySQLToElasticSearch()
        total = 0
        for table_struct in my_dump.structure:
            print("\n\n[+] Fetching resource for %s table." % \
            table_struct["table_name"])
            records = my_dump.get_data(table_struct)
            total += len(records)
            print("[+] Fetched %d records from %s." % (len(records), table_struct["table_name"]))
            for record in records:
                my_dump.push_to_elasticsearch(record, table_struct["table_name"])

        print("[+] Total records pushed - %d." % total)
    except KeyboardInterrupt:
        print("\n\n[+] CTRL-C pressed. Exiting!")


if __name__ == "__main__":
    main()