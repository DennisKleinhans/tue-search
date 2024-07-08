from datasets import load_dataset, disable_caching
from transformers import BertConfig
from preprocessing import PreprocessingModule
from training import TrainingModuleV2
import numpy as np


def create_glove_embedding_map():
    if pipeline_config.embedding_type == "glove":
        embedding_map = {}
        with open(pipeline_config.dataset_save_path+pipeline_config.glove_file, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = list(np.asarray(values[1:], dtype=float))
                except ValueError:
                    continue
                embedding_map[word] = coefs
    return embedding_map


if __name__ == '__main__':
    CONFIG_SAVE_PATH = "config/retrieval_system/"
    pipeline_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+"pipeline_config.json")
    training_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{pipeline_config.model}-training_config.json")
    model_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{pipeline_config.model}-model_config.json")

    PERFORM_RETRIEVAL = True # FF only


    # dataset loading
    disable_caching()  # needed for vocab generation!!!
    msmarco = load_dataset(
        "microsoft/ms_marco", "v2.1", split="train", verification_mode="no_checks"
    ).flatten()


    # tokenization + vocab creation
    PM = PreprocessingModule(
        training_config,
        model_config,
        pipeline_config
    )
    preprocessed_dataset, vocab = PM.execute(msmarco)

    # update params
    model_config.num_embeddings = len(vocab)                                                                    
    model_config.vocab_size = len(vocab)

    print("loading embedding map...")
    embed_map = None
    if pipeline_config.load_dataset_from_disk:
        print(" - skipped")
    else:
        embed_map = create_glove_embedding_map()
        print(" - done")

    TM = TrainingModuleV2(
        training_config,
        model_config,
        pipeline_config
    )
    TM.execute(preprocessed_dataset, embed_map)
