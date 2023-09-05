import os
from .utils.ilasp_task_generator_example import generate_examples
from .utils.ilasp_task_generator_hypothesis import get_hypothesis_space
from .utils.ilasp_task_generator_state import generate_state_statements
from .utils.ilasp_task_generator_symmetry_breaking import generate_symmetry_breaking_statements
from .utils.ilasp_task_generator_transition import generate_timestep_statements, generate_state_at_timestep_statements, generate_transition_statements


def generate_ilasp_task(num_states, accepting_state, rejecting_state, observables, goal_examples, neg_examples,
                        inc_examples, output_folder, output_filename, symmetry_breaking_method, max_disj_size,
                        learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                        prioritize_optimal_solutions, binary_folder_name):
    # statements will not be generated for the rejecting state if there are not deadend examples
    rejecting_state = None

    with open(os.path.join(output_folder, output_filename), 'w') as f:
        task = _generate_ilasp_task_str(num_states, accepting_state, rejecting_state, observables, goal_examples,
                                        neg_examples, inc_examples, output_folder, symmetry_breaking_method,
                                        max_disj_size, learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                                        prioritize_optimal_solutions, binary_folder_name)
        f.write(task)


def _generate_ilasp_task_str(num_states, accepting_state, rejecting_state, observables, goal_examples, neg_examples,
                             inc_examples, output_folder, symmetry_breaking_method, max_disj_size, learn_acyclic,
                             use_compressed_traces, avoid_learning_only_negative, prioritize_optimal_solutions, binary_folder_name):
    task = generate_state_statements(num_states, accepting_state, rejecting_state)
    task += generate_timestep_statements(goal_examples, neg_examples, inc_examples)
    task += _generate_edge_indices_facts(max_disj_size)
    task += generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state)
    task += generate_transition_statements(learn_acyclic, use_compressed_traces, avoid_learning_only_negative, prioritize_optimal_solutions)
    task += get_hypothesis_space(num_states, accepting_state, rejecting_state, observables, output_folder,
                                 symmetry_breaking_method, max_disj_size, learn_acyclic, binary_folder_name)

    if symmetry_breaking_method is not None:
        task += generate_symmetry_breaking_statements(num_states, accepting_state, rejecting_state, observables,
                                                      symmetry_breaking_method, max_disj_size, learn_acyclic)

    task += generate_examples(goal_examples, neg_examples, inc_examples)
    return task


def _generate_edge_indices_facts(max_disj_size):
    return "edge_id(1..%d).\n\n" % max_disj_size

if __name__ == "__main__":

    ilasp_task_filename = f"task_test"
    ilasp_folder = "/Users/leo/dev/phd/rm-cooperative-marl/experiments/buttons/agent_0/"

    observables = [
        "by", "br", "g"
    ]

    example1 = ('by', 'br', 'g')
    example2 = ('by', 'by', 'br', 'g')
    example3 = ('by', 'br', 'br', 'g')
    positive_examples = {
        example1,
        example2,
        example3
    }

    intersection = set(observables)
    # for e in positive_examples:
    #     intersection = intersection.intersection(set(e))

    incomplete_examples = set()
    for example in positive_examples:
        for i in range(len(example)-1):
            pre = example[:i+1]
            if pre not in positive_examples:
                incomplete_examples.add(pre)
            post = example[-i-1:]
            if post not in positive_examples:
                incomplete_examples.add(post)

    num_states = 3 + 2

    # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
    # can produce different hypothesis for the same set of examples but given in different order)
    generate_ilasp_task(
        num_states, 
        "u_acc",
        "u_rej", 
        observables, 
        sorted(positive_examples),
        sorted(set()), 
        sorted(incomplete_examples),
        ilasp_folder,
        ilasp_task_filename, 
        "bfs-alternative", # symmetry_breaking_method
        1, # max_disjunction_size
        False, # learn_acyclic_graph
        True, # use_compressed_traces
        True, # avoid_learning_only_negative 
        False, # prioritize_optimal_solutions 
        None # bin directory (ILASP is on PATH)
    )