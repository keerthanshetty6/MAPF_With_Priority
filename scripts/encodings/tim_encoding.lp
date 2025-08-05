#program makespan(horizon).
horizon(A, horizon) :- agent(A).
horizon(horizon).

#program sum_of_costs(d).
horizon(A, N+d) :- sp_length(A, N).               % Each agent’s horizon = shortest path length + d extra steps
horizon(M) :- M = #max { T : horizon(A, T) }.      % Global horizon = max individual horizon
bound(H+d) :- H = #sum { N, A : sp_length(A, N) }. % Total shortest-path length sum plus d extra moves

% Penalty definition for extra moves (beyond shortest path length)
penalty(A, N) :- sp_length(A, N+1), N >= 0.
penalty(A, T) :- sp_length(A, N), at(A, U, T), not goal(A, U), T >= N.
penalty(A, T) :- penalty(A, T+1), T >= 0.

% Sum-of-costs constraint: total extra actions used by all agents ≤ d
:- #sum { 1, A, T : penalty(A, T) } > B, bound(B).


#program mapf.
% Time steps from 1 up to the horizon for each agent and a global time axis
time(A, 1..T) :- horizon(A, T).
time(1..T)    :- horizon(T).

% Directions and movement dynamics
dir(up; down; left; right; wait).
delta(right, (X,Y), (X+1,Y)) :- edge((X,Y), (X+1,Y)).
delta(left,  (X,Y), (X-1,Y)) :- edge((X,Y), (X-1,Y)).
delta(up,    (X,Y), (X, Y+1)) :- edge((X,Y), (X, Y+1)).
delta(down,  (X,Y), (X, Y-1)) :- edge((X,Y), (X, Y-1)).
delta(wait,  (X,Y), (X, Y))   :- vertex((X,Y)).  % waiting means staying on the same vertex

% Initial positions at time 0
at(A, P, 0) :- start(A, P), agent(A).

% Movement choice: each agent either moves in one direction or waits at each time step T
{ moved(A, D, T) : dir(D) } 1 :- time(A, T).

% State update (movement and inertia)
at(A, V, T) :- moved(A, D, T), at(A, U, T-1), delta(D, U, V).
at(A, V, T) :- at(A, V, T-1), not moved(A, _, T), time(A, T).

% Collision constraints
% Vertex collision: no two agents at same V at time T:contentReference[oaicite:13]{index=13}
:- { at(A, V, T) } > 1, vertex(V), time(_, T).    
move(U, V, T) :- at(A, U, T-1), moved(A, D, T), delta(D, U, V), time(A, T).
% Swap collision: prevent two agents swapping positions:contentReference[oaicite:14]{index=14}
:- move(U, V, T), move(V, U, T), U < V.           

% Each agent must have exactly one position per time
% Inertia/Moved ensures at most one; this ensures at least one:contentReference[oaicite:15]{index=15}
:- { at(A, V, T) } != 1, agent(A), time(A, T).    

% Goal reachability (makes the instance unsatisfiable if any agent not at goal by its horizon)
% All agents must be at goal at time H (their horizon):contentReference[oaicite:16]{index=16}
:- goal(A, V), not at(A, V, H), horizon(A, H).    

#defined move/4.      % (Defined to accommodate heuristic references to move/4, though moved/3 is used for actual moves)
#defined moved/3.
#defined penalty/2.
#defined horizon/1.
#defined horizon/2.
#defined sp_length/2.
#defined bound/1.
#defined penalty_summary/3.
#defined vertex/1.
#defined edge/2.