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
To generate the necessary json files if they are not already present in the result folder run:
```bash
python build_complete_index.py
```

## Flask Backend 

To run the Flask backend, simply run:
 ```bash
 python src/middleware/app.py
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


## License
This release of TUESearch is licensed under the BSD 3-Clause license. Please refer to the LICENSE file for the terms of this license.


## Support
This release comes without any support, warranty or guarantee that your PC won't melt while training the retrieval system.
