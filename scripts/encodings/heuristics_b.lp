#program heuristics.
#heuristic move(A,U,V,T) : edge(U,V), time(T), priority(A, PA), priority(B, PB), move(B,_,V,T), A != B, PA > PB. [PA,level]
%#heuristic move(A,D,T) : at(A,U,T-1), delta(D,U,V), priority(A,PA), priority(B,PB), move(B,DB,T), at(B,UB,T-1), delta(DB,UB,V), A != B, PA > PB. [PA,level]
