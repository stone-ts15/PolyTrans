grammar ECQL;

expr: sql_expr | cypher_expr ;
sub_expr: '{' expr '}' ;

sql_expr: select_clause from_clause where_clause group_by_clause? order_by_clause? limit_clause? ;
select_clause: 'SELECT' proj_list ;
proj_list: item | item ',' proj_list ;
from_clause: 'FROM' table_list ;
table_list: table_def | table_def ',' table_list ;
table_def: VAR | VAR 'AS' VAR ;
where_clause: 'WHERE' predicates ;
predicates: predicate | predicate 'AND' predicates ;
predicate: item_or_subexpr cmp_symbol item_or_subexpr
         | func_expr
         ;
group_by_clause: 'GROUP' 'BY' var ;
order_by_clause: 'ORDER' 'BY' var order? ;
limit_clause: 'LIMIT' INT ;

cypher_expr: match_clause cypher_inner return_clause
           | match_clause cypher_inner return_clause order_by_clause limit_clause
           ;

cypher_inner: cypher_inner_clause cypher_inner | ;
cypher_inner_clause: match_clause | cypher_where_clause | with_clause ;

match_clause: 'MATCH' path_patterns ;
path_patterns: path_pattern | path_pattern ',' path_patterns ;
path_pattern: node_pattern | node_pattern rel_pattern node_pattern ;
node_pattern: '(' node ':' label ')' ;
rel_pattern: '-[' rel ':' label ']->' ;
cypher_where_clause: 'WHERE' predicates ;
with_clause: 'WITH' proj_list ;
return_clause: 'RETURN' vars ;

order: 'ASC' | 'DESC' ;
node: VAR ;
rel: VAR | ;
label: VAR ;
item_or_subexpr: item | sub_expr ;
item: var
    | var 'AS' VAR
    | func_expr
    | func_expr 'AS' VAR
    | STAR
    | literal
    ;
vars: var | var ',' vars ;
var: VAR | VAR '.' VAR ;
func_expr: func_name '(' func_input ')' ;
func_name: VAR ;
func_input: item | item ',' func_input ;
cmp_symbol: '=' | '>' | '>=' | '<' | '<=' | '<>' | 'IN' | '@>' | 'IS' | 'IS NOT' ;
literal: INT | STRING | FLOAT | 'NULL' ;
VAR: [a-zA-Z_][a-zA-Z0-9_]* ;
STAR: '*' ;
INT: [0-9]+ ;
STRING: '"' ~('"')* '"' | '\'' ~('\'')* '\'' ;
FLOAT: INT '.' INT ;


WS: [ \t\r\n]+ -> skip;
