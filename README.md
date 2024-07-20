# TUESearch

## Crawler
### Install playwright 
First install dependencies
```
pip install -r requirements.txt
```

Then install playwright
```
playwright install chromium
playwright install-deps chromium
```

### VSCode launch.json
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
### Run crawler
```
python src/crawler/crawler.py
```

## React Frontend

To run the React frontend, you need to install Node.js (https://nodejs.org). Then follow the steps below:

1. **Change to the project directory**: Navigate to the src/frontend directory.
    ```bash
    cd src/frontend
    ```
2. **Install dependencies**: Install the necessary dependencies with `npm` or `yarn`.
    ```bash
    npm install
    ```
    or
    ```bash
    yarn install
    ```
3. **Start web server**: Start the web server.
    ```bash
    npm start
    ```
    or
    ```bash
    yarn start
    ```
4. **Open frontend in browser**: Open your browser and navigate to `http://localhost:3000` to view the application.

## Indexing

Files to run to generate the necessary json files if they are not in the results folder:

1. **build_inverted_index.py**: 
   ```
   python ./src/indexing/build_inverted_index.py
   ```
2. **build_tf_and_idf.py**: 
   ```
   python ./src/indexing/build_tf_and_idf.py
   ```
3. **build_tf_idf.py**: 
   ```
   python ./src/indexing/build_tf_idf.py
   ```

## Flask Backend 

To run the Flask backend, please carry out the following steps:

1. **Change to the project directory**: Navigate to the src/middleware directory.
    ```bash
    cd src/middleware
    ```
2. **Start backend server**: Start the backend server.
    ```bash
    python app.py
    ```


## Retrieval System
To train the retrieval system on the [MSMARCO](https://huggingface.co/datasets/microsoft/ms_marco) dataset as described in the project report, run 
```bash
    python src\retrieval_system\logistic_regression\executable.py
```
This will train **and** evaluate the model using a 80/20 train/test split. 

You can manipulate the training process with the configuration files:
 - `config\retrieval_system\LR-training_config.json`
 - `config\retrieval_system\pipeline_config.json`

Note that currently only `pipeline_config.model = "LR"` (logistic regression) and `pipeline_config.embedding_type = "none"` (this disables the use of GloVe) are supported.
