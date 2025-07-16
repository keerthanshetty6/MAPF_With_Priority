"""
A simple MAPF solver based on clingo.
"""

import sys 
import timeit 
from collections import defaultdict #to create dictionaries with default values
from typing import Any, DefaultDict, List, Optional, Sequence, Tuple #Provides type hints for better code clarity and type checking.

from clingo.application import Application, ApplicationOptions, Flag, clingo_main  #Function(clingo_main) and classes to implement applications based on clingo.
from clingo.control import Control, Model #This module contains the Control class responsible for controling grounding and solving
from clingo.symbol import Function, Number, Symbol #Functions and classes for symbol manipulation

from cmapf import Objective, Problem, count_atoms


class PriorityMAPFApp(Application): #inherits from Application -> Clingo's base class for defining applications
    """
    A Multi-Agent Pathfinding (MAPF) solver.
    """

    def __init__(self):
        self._delta_or_horizon : Optional[int]      = None # optional delta/horizon value
        self._heuristic_strategy: str               = "No" # heuristic strategy, can be "No", "A", or "B"
        self._reach :Flag                           = Flag(True) # whether to compute reach atoms via cmapf or ASP
        self._costs :Flag                           = Flag(True) # whether to add per agent costs to models
        self._stats : dict                          = {"Time": {}} # a dictionary to gather statistics
        self._sp : DefaultDict[Symbol, Symbol]      = defaultdict(lambda: Number(0))  # a mapping from agents to their shortest path lengths
        self._penalties : List[Tuple[int, Symbol]]  = None # a list of literal and symbols for penalties
        self._objective : Objective                 = Objective.SUM_OF_COSTS # the objective to solve for (0 for sum_of_costs)
        self._objectives : int                      = 0 # the number of objectives given on the command line(there will be an error if there is more than one)
        self._finish : Flag                         = Flag(True)
        self._map_file: Optional[str]               = None
        self._scen_file: Optional[str]              = None
        self._agent_count: Optional[int]            = None
        self._compute_min_horizon: Flag             = Flag(False)

    def _parse_delta(self, value: str, objective: Objective) -> bool:
        """
         Parse and set the delta or horizon value (delta for Soc or horizon for makespan).

        Args:
            value (str): The value of delta or horizon --delta = number.
            objective (Objective): The optimization objective .

        Returns:
            bool: True if parsing is successful, False otherwise.
        """
        self._objective = objective
        self._objectives += 1

        if value == "auto":
            self._delta_or_horizon = None
            return True
        
        try:
            self._delta_or_horizon = int(value)
            return self._delta_or_horizon >= 0
        except ValueError:
            print(f"Invalid value for delta or horizon: {value}")
            return False
        
    def _parse_heuristic(self, value: str) -> bool:
        if value not in ["No", "A", "B"]:
            print(f"Invalid heuristic strategy: {value}. Must be one of No, A, B.")
            return False
        self._heuristic_strategy = value
        return True

    #callback function to update statistics. The step and accumulated statistics are passed as arguments.
    def _on_statistics(self, step: dict, accu: dict) -> None:
        """
        Add statistics.
        """
        stats : dict[str, dict]= {"PriorityMAPF": self._stats}
        step.update(stats)
        accu.update(stats)

    def _inject_instance(self, ctl: Control, map_file: str, scen_file: str, agent_count: int) -> None:
        """
        Inject map and scenario facts into Clingo backend.
        """
        
        with open(map_file, "r") as f:
            map_str = f.read().splitlines()[4:]  # skip header
        with open(scen_file, "r") as f:
            scen_lines = f.readlines()[1:agent_count+1]

        with ctl.backend() as bck:
            for row in range(len(map_str)):
                for col in range(len(map_str[row])):
                    if map_str[row][col] in ['.', 'E', 'S', 'T']:
                        pos = Function("", [Number(col), Number(row)])
                        a = bck.add_atom(Function("vertex", [pos]))
                        bck.add_rule([a])
                        if row > 0 and map_str[row-1][col] in ['.', 'E', 'S', 'G']:
                            x = Function("", [Number(col), Number(row)])
                            y = Function("", [Number(col), Number(row-1)])
                            a2 = bck.add_atom(Function("edge", [x, y]))
                            bck.add_rule([a2])
                            a3 = bck.add_atom(Function("edge", [y, x]))
                            bck.add_rule([a3])
                        if col > 0 and map_str[row][col-1] in ['.', 'E', 'S', 'G']:
                            x = Function("", [Number(col), Number(row)])
                            y = Function("", [Number(col-1), Number(row)])
                            a2 = bck.add_atom(Function("edge", [x, y]))
                            bck.add_rule([a2])
                            a3 = bck.add_atom(Function("edge", [y, x]))
                            bck.add_rule([a3])       

            for i, line in enumerate(scen_lines, start=1):
                parts = line.split()
                try:
                    sr, sc, gr, gc = int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
                except ValueError as e:
                    print(f"Failed to parse scenario line: {line.strip()} → {e}")
                    continue
                a_agent = bck.add_atom(Function("agent", [Number(i)]))
                a_start = bck.add_atom(Function("start", [Number(i), Function("", [Number(sr), Number(sc)])]))
                a_goal  = bck.add_atom(Function("goal", [Number(i), Function("", [Number(gr), Number(gc)])]))
                bck.add_rule([a_agent])
                bck.add_rule([a_start])
                bck.add_rule([a_goal])

    def register_options(self, options: ApplicationOptions) -> None:
        """
        Register MAPF options.
        """
        
        def parse_agents(value: str) -> bool:
            """Parse and validate agent count."""
            try:
                agent_count = int(value)
                if agent_count > 0:
                    self._agent_count = agent_count
                    return True
                else:
                    print(f"Error: Agent count must be positive, got {agent_count}")
                    return False
            except ValueError:
                print(f"Error: Invalid agent count '{value}', must be an integer")
                return False
        
        def parse_map_file(value: str) -> bool:
            """Parse and validate map file."""
            self._map_file = value
            return True
        
        def parse_scen_file(value: str) -> bool:
            """Parse and validate scenario file."""
            self._scen_file = value
            return True
        
        options.add(
            "PriorityMAPF",
            "delta",
            "set the delta value",
            lambda value: self._parse_delta(value, Objective.SUM_OF_COSTS),
        )
        options.add(
            "PriorityMAPF",
            "horizon",
            "set the horizon value",
            lambda value: self._parse_delta(value, Objective.MAKESPAN),
        )
        options.add_flag(
            "PriorityMAPF", "reach", "compute reachable positions with CMAPF", self._reach
        )
        options.add_flag(
            "PriorityMAPF", "show-costs", "add per agents costs to model", self._costs
        )
        options.add(
            "PriorityMAPF",
            "heuristic-strategy",
            "select heuristic strategy (No, A, B)",
            lambda value: self._parse_heuristic(value),
        )
        options.add(
            "PriorityMAPF",
            "map-file",
            "path to the map file",
            parse_map_file,
        )
        options.add(
            "PriorityMAPF",
            "scen-file",
            "path to the scenario file",
            parse_scen_file,
        )
        options.add(
            "PriorityMAPF",
            "agents",
            "number of agents",
            parse_agents,
        )
        options.add_flag(
            "PriorityMAPF", 
            "compute-min-horizon", 
            "compute and print min horizon and exit", 
            self._compute_min_horizon
)

    def print_model(self, model: Model, printer=None) -> None:
        # Suppress default output
        pass

    def validate_options(self) -> bool:
        """
        Validate options.
        """
        if self._objectives > 1:
            print("Error: either a delta value or a horizon should be passed, not both.")
            return False
        
        if self._heuristic_strategy not in ["No", "A", "B"]:
            print(f"Error: Invalid heuristic strategy: {self._heuristic_strategy}. Must be one of No, A, B.")
            return False

        if (self._map_file or self._scen_file) and self._agent_count is None:
            print("Error: If providing map/scen files, you must also specify --agents.")
            return False
        
        
        
        return True

    def _on_model(self, model: Model) -> None:
        """
        Extend the model with per-agent costs.

        Args:
            model (Model): The current model.
        """
        # precompute list of penalties
        if self._penalties is None:
            atoms = model.context.symbolic_atoms # all atoms in the logic program
            self._penalties = []
            for atom in atoms.by_signature("penalty", 2): #filters atoms with the signature penalty/2
                agent, _ = atom.symbol.arguments #agent value
                self._penalties.append((atom.literal, agent))

        # Calculate costs for each agent
        costs: DefaultDict[Symbol, int] = defaultdict(int)
        for literal, agent in self._penalties:
            if model.is_true(literal): #check if the literal is true in the current model
                costs[agent] += 1 #{agent:costs}
        # Extend the model with cost information, adding new symbols (facts) to the model {cost}

        total_cost=0
        max_horizon=0
        for agent,cost in costs.items():
            total_cost += cost #sum of all path taken
            max_horizon = max(max_horizon, cost) #max of agents path taken

        self._stats["Total Cost"] = total_cost
        self._stats["Max Horizon"] = max_horizon

        model.extend(
            [
                Function("penalty_summary", [agent, Function("shortest_path", [self._sp[agent]]),Function("Path_taken", [Number(cost)])])
                for agent, cost in costs.items()
            ]
        )


    def _load(self, ctl: Control, files: List[str], map_file: Optional[str]=None, scen_file: Optional[str]=None, agent_count: Optional[int]=None) -> Problem: #clingo_main creates a control object
        """
        Load instance and encoding and then extract the MAPF problem.
        """
        # load files
        start = timeit.default_timer()

        for file in files:
            print(file)
            ctl.load(file)
        if not files:
            print("No files provided, loading from map and scenario files.")

        self._stats["Time"]["Load"] = timeit.default_timer() - start

        # ground instance in base program
        start = timeit.default_timer()
        if map_file and scen_file and agent_count:
            self._inject_instance(ctl, map_file, scen_file, agent_count)

        ctl.ground()
        self._stats["Time"]["Ground Instance"] = timeit.default_timer() - start
  
        start = timeit.default_timer()
        problem = Problem(ctl)
        self._stats["Time"]["Extract Problem"] = timeit.default_timer() - start

        return problem

    def _prepare(self, ctl: Control, problem: Problem) -> Optional[Sequence[Tuple[str, Sequence[Symbol]]]]:
        """
        Prepare for grounding and return the necessary parts to ground.
        """
        # either compute or use given delta/horizon
        if self._delta_or_horizon is None:
            # Compute delta or horizon automatically
            start = timeit.default_timer()
            delta_or_horizon = problem.min_delta_or_horizon(self._objective) #always 1?
            self._stats["Time"]["Min Delta"] = timeit.default_timer() - start
        else:
            delta_or_horizon = self._delta_or_horizon

        if delta_or_horizon is None:
            return None # Problem is unsatisfiable

        # select program parts based on objective
        parts: List[Tuple[str, Sequence[Symbol]]] = [("mapf", [])]
        if self._objective == Objective.MAKESPAN:
            parts.append(("makespan", [Number(delta_or_horizon)])) #program makespan(horizon).
            self._stats["Horizon"] = delta_or_horizon 
        else:
            parts.append(("sum_of_costs", [Number(delta_or_horizon)]))  # program sum_of_costs(delta).
            self._stats["Delta"] = delta_or_horizon

        # select heuristic strategy
        if self._heuristic_strategy in ("A", "B"):
            parts.append(("heuristics", []))  # so the #program heuristics. section is grounded

        # always add shortest paths
        if not self._reach or self._objective == Objective.MAKESPAN:
            start = timeit.default_timer()
            if not problem.add_sp_length(ctl):
                parts = None
            self._stats["Time"]["Shortest Path"] = timeit.default_timer() - start
          

        if self._reach:
            # reachability computation via C++ has been requested
            start = timeit.default_timer()
            if not problem.add_reachable(ctl, self._objective, delta_or_horizon): #sp_length is calculated within compute_reach
                parts = None
            self._stats["Time"]["Reachable"] = timeit.default_timer() - start
        else:
            # reachability computation via ASP has been requested
            parts.append(("reach", []))

        return parts

    def _ground(self, ctl: Control, parts: Optional[Sequence[Tuple[str, Sequence[Symbol]]]]) -> None:
        """
        Ground the MAPF encoding.
        """
        start = timeit.default_timer()
        
        if parts is not None:
            # ground the encoding
            ctl.ground(parts)
            self._stats["Reachable"] = count_atoms(ctl.symbolic_atoms, "reach", 3)

            # compute the minimum cost as the sum of the shortest path lengths
            min_cost = 0
            min_horizon = 0
            for atom in ctl.symbolic_atoms.by_signature("sp_length", 2):
                agent, length = atom.symbol.arguments
                self._sp[agent] = length
                min_cost += length.number #sum of all sp
                min_horizon = max(min_horizon, length.number) #max of agents sp

            self._stats["Min Cost"] = min_cost
            self._stats["Min Horizon"] = min_horizon
        else:
            # make the problem unsatisfiable avoiding grounding, when some agent can not reach its goal
            with ctl.backend() as bck:
                bck.add_rule([])
        self._stats["Time"]["Ground Encoding"] = timeit.default_timer() - start

    def _solve(self, ctl: Control):
        """
        Solve the MAPF problem.
        """
        start = timeit.default_timer()
        kwargs: dict = {"on_statistics": self._on_statistics}
        if self._costs:
            kwargs["on_model"] = self._on_model # add cost symbol
        result  = ctl.solve(**kwargs)
        
        self._stats["Time"]["Solve Encoding"] = timeit.default_timer() - start
        return result

    def _on_finish(self, result) -> None:
        """
        Handle the completion of the solving process, controlled by the _on_finish flag.
        """
        
        if not self._finish:
            return  # If the flag is False, skip this method.

        if result.satisfiable:
            print("The problem is satisfiable!")

        elif result.unsatisfiable:
            print("The problem is unsatisfiable.")
        else:
            print("The solving process returned an unknown result.")
        
        # Print out the time statistics here
        print("Statistics:")
        for category, times in self._stats["Time"].items():
            print(f"{category}: {times:.7f}")
        print("Finish")


    def main(self, ctl: Control, files) -> None:
        """
        The main function of the application.
        """
        if self._compute_min_horizon:
            # just load and ground sp_length
            problem = self._load(ctl, [], map_file=self._map_file, scen_file=self._scen_file, agent_count=self._agent_count)
            problem.add_sp_length(ctl)
            ctl.ground()
            min_horizon = 0
            for atom in ctl.symbolic_atoms.by_signature("sp_length", 2):
                _, length = atom.symbol.arguments
                print(length)
                min_horizon = max(min_horizon, length.number)
            print(f"Min Horizon: {min_horizon}")
            return
        
        problem = self._load(ctl,files,map_file=self._map_file,scen_file=self._scen_file,agent_count=self._agent_count)
        parts = self._prepare(ctl, problem)
        self._ground(ctl, parts)
        result = self._solve(ctl)
        self._on_finish(result)

if __name__ == "__main__":
    clingo_main(PriorityMAPFApp(), sys.argv[1:])
