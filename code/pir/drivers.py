import psycopg2
import py2neo
import pyArango
import re

import pyArango.connection
import pyArango.database

from db_config import *

_arango_conn = None
_pg_conn = None
_neo_conn = None

def get_rel_conn():
    global _pg_conn
    if not _pg_conn:
        _pg_conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, database=PG_DBNAME, user=PG_USER, password=PG_PASSWORD)
        _pg_conn.autocommit = True
    return _pg_conn

def get_graph_conn(force=False):
    global _neo_conn
    if force or not _neo_conn:
        _neo_conn = py2neo.Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo_conn

def get_arango_conn():
    global _arango_conn
    if not _arango_conn:
        conn = pyArango.connection.Connection(arangoURL=ARANGO_URL, username=ARANGO_USER, password=ARANGO_PASSWORD)
        _arango_conn = conn[ARANGO_DBNAME]
    return _arango_conn

class Solver:
    def __init__(self):
        self.result = None

    def format_value(self, v):
        if isinstance(v, str):
            return "'" + v + "'"
        else:
            return str(v)

    def reduce(self, rtype):
        if rtype == 'count':
            return self.reduce_count()
        elif rtype == 'str':
            return self.reduce_str()
        elif rtype == 'row':
            return self.reduce_row()
        else:
            print('Cannot recognize scope')
            return '()'

    def reduce_count(self):
        return len(self.result)

    def reduce_str(self):
        return ','.join(self.format_value(list(r.values())[0]) for r in self.result) if self.result else '-1'
    
    def reduce_row(self):
        def format(r):
            if isinstance(r, dict):
                return ','.join(str(v) for v in r.values())
            elif isinstance(r, tuple):
                return ','.join(str(v) for v in r)
            else:
                return str(r)
        return [format(r) for r in self.result]

    def execute(self, q):
        pass

class ArangoDBSolver(Solver):
    def __init__(self):
        super().__init__()

    def execute(self, q):
        aql_query = get_arango_conn().AQLQuery(q, rawResults=True)
        self.result = [r for r in aql_query]


class PostgresSolver(Solver):
    def __init__(self):
        super().__init__()

    def execute(self, q):
        cursor = get_rel_conn().cursor()
        cursor.execute(q)
        self.result = cursor.fetchall()

    def reduce_str(self):
        return ','.join(self.format_value(r[0]) for r in self.result) if self.result else '-1'

class Neo4jSolver(Solver):
    def __init__(self):
        super().__init__()
        self.label_maps = {
            'IMDB': ['Movie', 'TvSeries', 'Actor', 'Cast', 'Company', 'Copyright', 'Director', 'DirectedBy', 'Producer', 'MadeBy', 'Writer', 'WrittenBy']
        }

    def execute(self, q):
        fq = ''
        while True:
            m = re.search(r':([a-zA-Z]+)', q)
            if m:
                label = m.group(1)
                ds = ''
                for dataset_name, labels in self.label_maps.items():
                    if label in labels:
                        ds = dataset_name
                        break
                fq += q[:m.span()[0]] + ':' + ds + m.group(1)
                q = q[m.span()[1]:]
            else:
                fq += q
                break

        self.result = get_graph_conn().run(fq).data()

def fetch(q, reduce='str'):
    if q.startswith('FOR') or q.startswith('LET') or q.startswith('RETURN'):
        solver = ArangoDBSolver()
    elif q.startswith('SELECT'):
        solver = PostgresSolver()
    elif q.startswith('MATCH'):
        solver = Neo4jSolver()
    else:
        print('Cannot recognize scope')
        return '()'
    
    solver.execute(q)
    return solver.reduce(reduce)
