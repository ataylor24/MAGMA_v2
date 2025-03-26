import data_generation_utils.magma_samplers as smp
import data_generation_utils.clrs_text_utils as clrs_text_utils
import data_generation_utils.data_utils as data_utils
from data_generation_utils.data_utils import iterate_sampler, hash_edgelist
from data_generation_utils.translators import translate_inputs, translate_hints, translate_outputs
from data_generation_utils.data_writer import write_data
from globals import ( 
    FORMATTED_ALGORITHMS, 
    TRAIN_TEST_SPLIT, 
    _OOD_TRAIN_LENGTHS,
    _OOD_EVAL_LENGTHS
)


def sample_data(args):
    clrs_training_data = {}
    clrs_validation_data = {}
    clrs_testing_data = {}

    clrs_text_training_data = {}
    clrs_text_validation_data = {}
    clrs_text_testing_data = {}

    clrs_text_no_hints_training_data = {}
    clrs_text_no_hints_validation_data = {}
    clrs_text_no_hints_testing_data = {}

    magma_training_data = {}
    magma_validation_data = {}
    magma_testing_data = {}

    if args.ood_generation:
        (clrs_data_dir, clrs_text_data_dir,
            clrs_text_no_hints_data_dir, dict_magma_data_dir) = data_utils.resolve_output_dirs(
                args.output_dir, args.output_formats
            )
        algorithms = args.algorithm if args.algorithm != "all" else FORMATTED_ALGORITHMS
        
        training_instances = TRAIN_TEST_SPLIT[-1][0]
        evaluation_instances = TRAIN_TEST_SPLIT[-1][1]

        data_spec = {"train": {}, "test": {}}
        num_training_samples = data_utils.count_random_selections(_OOD_TRAIN_LENGTHS, training_instances)
        
        valid_train_idx = 0
        valid_eval_idx = 0
        valid_test_idx = 0
        
        for algorithm in algorithms:
            for i, graph_size in enumerate(_OOD_TRAIN_LENGTHS):
                num_samples = num_training_samples[i]
                data_spec["train"][graph_size] = num_samples
                unique_graphs = set()

                data_smp, spec = smp.build_sampler(algorithm, num_samples=-1, length=graph_size, mixed_data=True, seed=args.seed)
                data_smp_iter = iterate_sampler(data_smp, batch_size=1)

                train_counter = 0
                while train_counter < num_samples:
                    train_sample = next(data_smp_iter)
                    inputs = translate_inputs(algorithm, train_sample.features.inputs)
                    edgelist_hash = hash_edgelist(inputs[1])
                    if edgelist_hash in unique_graphs:
                        continue

                    if algorithm in ["floyd_warshall", "dijkstra", "mst_prim", "dfs"]:
                        hints, final_d = translate_hints(algorithm, args.neg_edges, set(inputs[1]),
                                                        train_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, train_sample.outputs, final_d)
                    else:
                        hints = translate_hints(algorithm, args.neg_edges, set(inputs[1]),
                                                train_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, train_sample.outputs)

                    clrs_training_data[valid_train_idx] = train_sample
                    clrs_text_training_data[valid_train_idx] = clrs_text_utils.clrs_text_sample(algorithm, train_sample, use_hints=True)
                    clrs_text_no_hints_training_data[valid_train_idx] = clrs_text_utils.clrs_text_sample(algorithm, train_sample, use_hints=False)
                    magma_training_data[valid_train_idx] = {"inputs": inputs, "hints": hints, "outputs": outputs}

                    unique_graphs.add(edgelist_hash)
                    valid_train_idx += 1
                    train_counter += 1                

            # Process evaluation data for OOD generation
            num_eval_samples = data_utils.count_random_selections(_OOD_EVAL_LENGTHS, evaluation_instances)
            for i, graph_size in enumerate(_OOD_EVAL_LENGTHS):
                num_samples = num_eval_samples[i]
                data_spec["test"][graph_size] = num_samples
                unique_graphs = set()

                data_smp, spec = smp.build_sampler(algorithm, num_samples=-1, length=graph_size, seed=args.seed)
                data_smp_iter = iterate_sampler(data_smp, batch_size=1)

                eval_counter = 0
                while eval_counter < num_samples:
                    test_sample = next(data_smp_iter)
                    inputs = translate_inputs(algorithm, test_sample.features.inputs)
                    edgelist_hash = hash_edgelist(inputs[1])
                    if edgelist_hash in unique_graphs:
                        continue

                    if algorithm in ["floyd_warshall", "dijkstra", "mst_prim", "dfs"]:
                        hints, final_d = translate_hints(algorithm, args.neg_edges, set(inputs[1]),
                                                        test_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, test_sample.outputs, final_d)
                    elif algorithm == "bfs":
                        hints = translate_hints(algorithm, args.neg_edges, set(inputs[1]), test_sample.features.hints)
                        outputs = translate_outputs(algorithm, test_sample.outputs)
                    else:
                        hints = translate_hints(algorithm, args.neg_edges, set(inputs[0]),
                                                test_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, test_sample.outputs)

                    if eval_counter < num_samples // 2:
                        clrs_validation_data[valid_eval_idx] = test_sample
                        clrs_text_validation_data[valid_eval_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=True)
                        clrs_text_no_hints_validation_data[valid_eval_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=False)
                        magma_validation_data[valid_eval_idx] = {"inputs": inputs, "hints": hints, "outputs": outputs}
                        valid_eval_idx += 1
                    else:
                        clrs_testing_data[valid_test_idx] = test_sample
                        clrs_text_testing_data[valid_test_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=True)
                        clrs_text_no_hints_testing_data[valid_test_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=False)
                        magma_testing_data[valid_test_idx] = {"inputs": inputs, "hints": hints, "outputs": outputs}
                        valid_test_idx += 1
                    unique_graphs.add(edgelist_hash)
                    eval_counter += 1  
                             
        print("Sampling complete for OOD")
        print("---Data Spec---")
        print(data_spec)
        print("---------------")

    else:
        algorithms = args.algorithm if args.algorithm != "all" else FORMATTED_ALGORITHMS
        
        for algorithm in algorithms:
            graph_sizes = [int(gs) for gs in args.graph_sizes]
            for graph_size in graph_sizes:
                unique_graphs = set()
                (clrs_data_dir, clrs_text_data_dir,
                clrs_text_no_hints_data_dir, dict_magma_data_dir) = data_utils.resolve_output_dirs(
                    args.output_dir, args.output_formats
                )
                training_instances, evaluation_instances = TRAIN_TEST_SPLIT.get(
                    graph_size, (args.train_test_split[0], args.train_test_split[1])
                )
                data_smp, spec = smp.build_sampler(algorithm, num_samples=-1, length=graph_size, seed=args.seed)
                data_smp_iter = iterate_sampler(data_smp, batch_size=1)

                valid_train_idx = 0
                while valid_train_idx < training_instances:
                    train_sample = next(data_smp_iter)
                    inputs = translate_inputs(algorithm, train_sample.features.inputs)
                    edgelist_hash = hash_edgelist(inputs[1])
                    if edgelist_hash in unique_graphs:
                        continue

                    if algorithm in ["floyd_warshall", "dijkstra", "mst_prim", "dfs"]:
                        hints, final_d = translate_hints(algorithm, args.neg_edges, set(inputs[1]),
                                                        train_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, train_sample.outputs, final_d)
                    else:
                        hints = translate_hints(algorithm, args.neg_edges, set(inputs[1]),
                                                train_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, train_sample.outputs)

                    clrs_training_data[valid_train_idx] = train_sample
                    clrs_text_training_data[valid_train_idx] = clrs_text_utils.clrs_text_sample(algorithm, train_sample, use_hints=True)
                    clrs_text_no_hints_training_data[valid_train_idx] = clrs_text_utils.clrs_text_sample(algorithm, train_sample, use_hints=False)
                    magma_training_data[valid_train_idx] = {"inputs": inputs, "hints": hints, "outputs": outputs}

                    unique_graphs.add(edgelist_hash)
                    valid_train_idx += 1

                valid_eval_idx = 0
                while valid_eval_idx < evaluation_instances:
                    test_sample = next(data_smp_iter)
                    inputs = translate_inputs(algorithm, test_sample.features.inputs)
                    edgelist_hash = hash_edgelist(inputs[1])
                    if edgelist_hash in unique_graphs:
                        continue

                    if algorithm in ["floyd_warshall", "dijkstra", "mst_prim", "dfs"]:
                        hints, final_d = translate_hints(algorithm, args.neg_edges, set(inputs[1]),
                                                        test_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, test_sample.outputs, final_d)
                    elif algorithm == "bfs":
                        hints = translate_hints(algorithm, args.neg_edges, set(inputs[1]), test_sample.features.hints)
                        outputs = translate_outputs(algorithm, test_sample.outputs)
                    else:
                        hints = translate_hints(algorithm, args.neg_edges, set(inputs[0]),
                                                test_sample.features.hints, source=inputs[2])
                        outputs = translate_outputs(algorithm, test_sample.outputs)

                    if valid_eval_idx < evaluation_instances // 2:
                        clrs_validation_data[valid_eval_idx] = test_sample
                        clrs_text_validation_data[valid_eval_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=True)
                        clrs_text_no_hints_validation_data[valid_eval_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=False)
                        magma_validation_data[valid_eval_idx] = {"inputs": inputs, "hints": hints, "outputs": outputs}
                    else:
                        test_idx = valid_eval_idx % (evaluation_instances // 2)
                        clrs_testing_data[test_idx] = test_sample
                        clrs_text_testing_data[test_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=True)
                        clrs_text_no_hints_testing_data[test_idx] = clrs_text_utils.clrs_text_sample(algorithm, test_sample, use_hints=False)
                        magma_testing_data[test_idx] = {"inputs": inputs, "hints": hints, "outputs": outputs}

                    unique_graphs.add(edgelist_hash)
                    valid_eval_idx += 1

            print(f"Sampling complete for graph size: {graph_size}")

    write_data(args,
               clrs_data_dir,
               clrs_text_data_dir,
               clrs_text_no_hints_data_dir,
               dict_magma_data_dir,
               clrs_training_data,
               clrs_validation_data,
               clrs_testing_data,
               clrs_text_training_data,
               clrs_text_validation_data,
               clrs_text_testing_data,
               clrs_text_no_hints_training_data,
               clrs_text_no_hints_validation_data,
               clrs_text_no_hints_testing_data,
               magma_training_data,
               magma_validation_data,
               magma_testing_data)

def main():
    args = data_utils.parse_args()
    sample_data(args)

if __name__ == "__main__":
    main()