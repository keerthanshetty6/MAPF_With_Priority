%% setup the makespan based encoding
#program makespan(horizon).
horizon(A,horizon) :- agent(A).
horizon(horizon).

%% setup the sum of costs based encoding
#program sum_of_costs(d).
horizon(A,N+d) :- sp_length(A,N). %max per agent
horizon(M) :- M = #max { N: horizon(A,N) }.
bound(H+d) :- H = #sum{T,A : sp_length(A,T)}.

penalty(A,N) :- sp_length(A,N+1), N >= 0.
penalty(A,T) :- sp_length(A,N), at(A,U,T), not goal(A,U), T >= N.
penalty(A,T) :- penalty(A,T+1), T >= 0.

:- #sum{ 1,A,T : penalty(A,T) } > B, bound(B).

#program mapf.
time(A,1..N) :- horizon(A,N).
time(1..N) :- horizon(N).

% Generate valid moves
{ move(A,U,V,T) : edge(U,V), reach(A,V,T) } 1 :- reach(A,U,T-1).

% Define
at(A,U,0) :- start(A,U).
at(A,V,T) :- move(A,_,V,T).
at(A,U,T) :- at(A,U,T-1), not move(A,U,_,T),reach(A,U,T).

%test
%can only move from current position
:- move(A,U,_,T), not at(A,U,T-1). 

%only one agent can occupy vertex U at time T - no 2 agent share the same vertex (vertex conflicts)
:- {at(A,U,T)} > 1, vertex(U), time(T). 

%swap constraints
:- move(_,U,V,T), move(_,V,U,T), U<V.

%follow constraints
:- at(A,U,T), at(B,U,T+1), A!=B, m=fc. 

% given a time agent A can be at only one vertex it , no need to have a at for higher time T 
:- {at(A,U,T)} != 1, time(A,T). 

%never at a non rechable position
%:- at(A,V,T), not reach(A,V,T ).

%check
:- goal(A,U), horizon(A,T), not at(A,U,T).


#defined penalty/2.
#defined move/4.
#defined horizon/2.
#defined sp_length/2.
#defined bound/1.
#defined penalty_summary/3.
#defined reach/3.
#defined delta/1.
#defined vertex/1.
#defined edge/2.