from datasets import load_dataset, Dataset
import numpy as np


def prepare_msmacro_dataset(train_size=0.85, drop_idx=None, tokenizer_model="bert-base-uncased"):
    print("Preparing dataset...")
    msmacro = load_dataset("microsoft/ms_marco", "v2.1", split="train", verification_mode="no_checks")

    if drop_idx is not None:
        msmacro = Dataset.from_dict(msmacro[:drop_idx])

    breakpoint = int(np.ceil(len(msmacro)*train_size))
    msmacro_train = Dataset.from_dict(msmacro[:breakpoint])
    msmacro_test = Dataset.from_dict(msmacro[breakpoint:])
    print(f"train size: {len(msmacro_train)}, test size: {len(msmacro_test)}")
    print(" - done")

    return msmacro_train, msmacro_test