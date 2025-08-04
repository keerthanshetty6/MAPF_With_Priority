dir(up;down;left;right).
dir(wait).
delta(right,(X,Y),(X+1,Y)) :- edge((X,Y),(X+1,Y)).
delta(left,(X,Y),(X-1,Y)) :- edge((X,Y),(X-1,Y)).
delta(up,(X,Y),(X,Y+1)) :- edge((X,Y),(X,Y+1)).
delta(down,(X,Y),(X,Y-1)) :- edge((X,Y),(X,Y-1)).
delta(wait,(X,Y),(X,Y)) :- vertex((X,Y)).

at(A,P,0) :- start(A,P), agent(A).

% if using asp to calculate reachability
{ moved(A,D,T) : dir(D)} 1 :- time(A,T).
% - move ------------------------------------------------------------------
at(A,V,T) :- moved(A,D,T), at(A,U,T-1), delta(D,U,V).


% - inertia ---------------------------------------------------------------
at(A,V,T) :- at(A,V,T-1), not moved(A,_,T), time(A,T).

% - vertex collision ------------------------------------------------------
:- { at(A,V,T) } > 1, vertex(V), time(_,T).
% % swap conflict
move(U,V,T) :- at(A,U,T-1), moved(A,D,T), delta(D,U,V), time(A,T).
:- move(U,V,T), move(V,U,T), U < V.
% - auxiliaries -------------------------------------------
:- { at(A,V,T) } != 1, agent(A), time(A,T).
% - query -----------------------------------------------------------------
:- goal(A,V), not at(A,V,T), horizon(A,T).