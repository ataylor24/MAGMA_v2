import os
from datasets import Dataset, DatasetDict
from . import data_utils

def write_data(args,
               clrs_data_dir,
               clrs_text_data_dir,
               clrs_text_no_hints_data_dir,
               dict_llm_data_dir,
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
               magma_testing_data):
    """Write out data in various formats for CLRS and LLM tasks."""
    # Write CLRS data
    for split_data, filename in zip(
            [clrs_training_data, clrs_validation_data, clrs_testing_data],
            ["training.pkl", "validation.pkl", "testing.pkl"]):
        data_utils.write_clrs_format(os.path.join(clrs_data_dir, filename), split_data)

    # Write CLRS Text data (with hints)
    dataset = DatasetDict({
        "train": Dataset.from_list(data_utils.write_clrs_chat_format(clrs_text_training_data)),
        "test": Dataset.from_list(data_utils.write_clrs_chat_format(clrs_text_validation_data)),
        "evaluation": Dataset.from_list(data_utils.write_clrs_chat_format(clrs_text_testing_data))
    })

    dataset.save_to_disk(clrs_text_data_dir)
    for split, split_data in [("training", clrs_text_training_data),
                              ("test", clrs_text_validation_data),
                              ("evaluation", clrs_text_testing_data)]:
        data_utils.write_json(os.path.join(clrs_text_data_dir, f"{split}.json"),
                              data_utils.write_clrs_chat_format(split_data))

    # Write CLRS Text data (without hints)
    dataset = DatasetDict({
        "train": Dataset.from_list(data_utils.write_clrs_chat_format(clrs_text_no_hints_training_data)),
        "test": Dataset.from_list(data_utils.write_clrs_chat_format(clrs_text_no_hints_validation_data)),
        "evaluation": Dataset.from_list(data_utils.write_clrs_chat_format(clrs_text_no_hints_testing_data))
    })
    dataset.save_to_disk(clrs_text_no_hints_data_dir)
    for split, split_data in [("training", clrs_text_no_hints_training_data),
                              ("test", clrs_text_no_hints_validation_data),
                              ("evaluation", clrs_text_no_hints_testing_data)]:
        data_utils.write_json(os.path.join(clrs_text_no_hints_data_dir, f"{split}.json"),
                              data_utils.write_clrs_chat_format(split_data))

    # Write LLM data
    for output_format in args.output_formats:
        llm_data_dir = dict_llm_data_dir[output_format]
        if output_format in data_utils.OUTPUT_FORMATS:
            for reasoning_type in data_utils.REASONING_STRATEGIES:
                for reasoning_strategy in data_utils.REASONING_STRATEGIES[reasoning_type]:
                    dataset = DatasetDict({
                        "train": Dataset.from_list(data_utils.write_chat_format(reasoning_type, reasoning_strategy, "training", magma_training_data)),
                        "test": Dataset.from_list(data_utils.write_chat_format(reasoning_type, reasoning_strategy, "evaluation", magma_validation_data)),
                        "evaluation": Dataset.from_list(data_utils.write_chat_format(reasoning_type, reasoning_strategy, "evaluation", magma_testing_data))
                    })
                    
                    outfile = os.path.join(llm_data_dir, reasoning_strategy)
                    
                    dataset.save_to_disk(outfile)
                    for split, split_data in [("train", magma_training_data),
                                                ("test", magma_validation_data),
                                                ("evaluation", magma_testing_data)]:
                        print(os.path.join(outfile, f"{split}.json"))
                        data_utils.write_json(os.path.join(outfile, f"{split}.json"),
                                                data_utils.write_chat_format(reasoning_type, reasoning_strategy, split, split_data))
        else:
            raise NotImplementedError(f"Output format {output_format} has not been implemented.")