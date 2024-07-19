# modern-search-engines

# Crawler
## Install playwright 
First install dependencies
```
pip install -r requirements.txt
```

Then install playwright
```
playwright install chromium
playwright install-deps chromium
```

## VSCode launch.json
```
{
   
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Website crawler",
            "type": "python",
            "request": "launch",
            "program": "src/crawler/crawler.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/modern-serach-engines/",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/modern-serach-engines/"
            }
        }
    ]
}
```

## Run crawler
```
python src/crawler/crawler.py
```

# Retrieval System
To train the retrieval system on the [MSMARCO](https://huggingface.co/datasets/microsoft/ms_marco) dataset as described in the project report, run `src\retrieval_system\logistic_regression\executable.py`. This will train **and** evaluate the model using a 80/20 train/test split. 

You can manipulate the training process with the configuration files `config\retrieval_system\LR-training_config.json` and `config\retrieval_system\pipeline_config.json`. Note that currently only `pipeline_config.model = "LR"` (logistic regression) and `pipeline_config.embedding_type = "none"` (this disables the use of GloVe) are supported.