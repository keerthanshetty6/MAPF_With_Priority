#program heuristics.
#heuristic move(A,_,V,T) : priority(A,P) . [P,level]
%#heuristic move(A,U,V,T) : edge(U,V), time(T), priority(A,P) . [P,level]