from ....reward_machine import RewardMachine
from ..ilasp_common import N_TRANSITION_STR, CONNECTED_STR
from ..task_parser.ilasp_parser_utils import parse_edge_rule, parse_negative_transition_rule


def parse_ilasp_solutions(ilasp_learnt_filename):
    with open(ilasp_learnt_filename) as f:
        rm = RewardMachine()
        edges = {}
        for line in f:
            line = line.strip()
            if line.startswith(N_TRANSITION_STR):
                parsed_transition = parse_negative_transition_rule(line)
                current_edge = ((parsed_transition.src, parsed_transition.dst), parsed_transition.edge)
                if current_edge not in edges:
                    edges[current_edge] = []
                for pos_fluent in parsed_transition.pos:
                    edges[current_edge].append("~" + pos_fluent)
                for neg_fluent in parsed_transition.neg:
                    edges[current_edge].append(neg_fluent)
            elif line.startswith(CONNECTED_STR):
                parsed_edge = parse_edge_rule(line)
                current_edge = ((parsed_edge.src, parsed_edge.dst), parsed_edge.edge)
                if current_edge not in edges:
                    edges[current_edge] = []

        for edge in edges:
            from_state, to_state = edge[0]
            rm.add_states([from_state, to_state])
            rm.add_transition(from_state, to_state, tuple(edges[edge]))

        return rm





