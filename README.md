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

To start the React frontend, please carry out the following steps:

1. **Change to the project directory**: Navigate to the src/frontend directory.
    ```bash
    cd src/frontend
    ```
2. **Install dependencies**: Install the necessary dependencies with `npm` or `yarn`.
    ```bash
    npm install
    ```
    oder
    ```bash
    yarn install
    ```
3. **Start web server**: Start the web server.
    ```bash
    npm start
    ```
    oder
    ```bash
    yarn start
    ```
4. **Open frontend in browser**: Open your browser and navigate to `http://localhost:3000` to view the application.

## Flask Backend 

To start the Flask backend, please carry out the following steps:

1. **Change to the project directory**: Navigate to the src/middleware directory.
    ```bash
    cd src/middleware
    ```
2. **Start backend server**: Start the backend server.
    ```bash
    python app.py
    ```
