from datasets import load_dataset, Dataset
import numpy as np


def prepare_msmarco_dataset(train_size=0.85, drop_idx=None):
    print("Preparing dataset...")
    msmarco = load_dataset("microsoft/ms_marco", "v2.1", split="train", verification_mode="no_checks")

    if drop_idx is not None:
        msmarco = Dataset.from_dict(msmarco[:drop_idx])

    breakpoint = int(np.ceil(len(msmarco)*train_size))
    msmarco_train = Dataset.from_dict(msmarco[:breakpoint])
    msmarco_test = Dataset.from_dict(msmarco[breakpoint:])
    print(f"train size: {len(msmarco_train)}, test size: {len(msmarco_test)}")
    print(" - done")

    return msmarco_train, msmarco_test