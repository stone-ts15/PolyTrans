grammar AQL;

expr: let_clause expr
    | for_clause expr
    | traversal_clause expr
    | filter_clause expr
    | sort_clause expr
    | limit_clause expr
    | aggregate_clause expr
    | return_clause
    ;
let_clause: 'LET' VAR '=' '(' expr ')' ;
for_clause: 'FOR' doc 'IN' collection ;
traversal_clause: 'FOR' doc ( ',' edge )? 'IN' depth direction doc collection ;
filter_clause: 'FILTER' predicates ;
sort_clause: 'SORT' var order? ;
limit_clause: 'LIMIT' limit ;
aggregate_clause: 'COLLECT' 'WITH' 'COUNT' 'INTO' doc
                | 'COLLECT' assign 'WITH' 'COUNT' 'INTO' doc
                | 'COLLECT' 'AGGREGATE' VAR '=' func_expr
                ;
return_clause: 'RETURN' var
             | 'RETURN' tuple_def
             | 'RETURN' obj_def
             ;

predicates: predicate | predicate 'AND' predicates ;
predicate: item cmp_symbol item
         | func_expr
         ;
func_expr: func_name '(' func_input ')' ;
func_input: item
         | item ',' func_input
         ;

assign: var '=' item ;
tuple_def: '{' vars '}' ;
obj_def: '{' kv_pairs '}' ;
kv_pairs: VAR ':' item
        | VAR ':' item ',' kv_pairs
        ;

vars: VAR | VAR ',' vars ;
var: VAR | VAR '.' VAR ;
varexpr: var '[' INT ']' ;
doc: VAR ;
edge: VAR ;
collection: VAR ;
depth: INT ;
direction: 'INBOUND' | 'OUTBOUND' ;
order: 'ASC' | 'DESC' ;
limit: INT ;
func_name: VAR ;

cmp_symbol: '==' | '>' | '>=' | '<' | '<=' | '!=' | 'IN' ;
item: var | varexpr | value | func_expr ;
value: INT | STRING | FLOAT ;

VAR: [a-zA-Z_][a-zA-Z0-9_]*;
INT: [0-9]+ ;
STRING: '"' ~('"')* '"' | '\'' ~('\'')* '\'' ;
FLOAT: INT '.' INT ;

WS: [ \t\r\n]+ -> skip;