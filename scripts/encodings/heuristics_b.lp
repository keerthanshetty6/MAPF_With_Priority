#program heuristics.
#heuristic move(A,_,V,T) : priority(A, PA), priority(B, PB), move(B,_,V,T), A != B, PA > PB. [PA,level]
%#heuristic move(A,U,V,T) : edge(U,V), time(T), priority(A,PA), priority(B,PB), move(B,_,V,T), A != B, PA > PB. [PA,level]