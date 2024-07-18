# TUESearch

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

### React-Frontend starten

Um das React-Frontend zu starten, führen Sie bitte die folgenden Schritte aus:

1. **In das Projektverzeichnis wechseln**: Navigieren Sie in das Verzeichnis src/frontend.
    ```bash
    cd src/frontend
    ```
2. **Abhängigkeiten installieren**: Installieren Sie die notwendigen Abhängigkeiten mit `npm` oder `yarn`.
    ```bash
    npm install
    ```
    oder
    ```bash
    yarn install
    ```
3. **Webserver starten**: Starten Sie den Webserver.
    ```bash
    npm start
    ```
    oder
    ```bash
    yarn start
    ```
4. **Frontend im Browser öffnen**: Öffnen Sie Ihren Browser und navigieren Sie zu `http://localhost:3000`, um die Anwendung zu sehen.

### Flask Backend starten 

Um das Flask Backend zu starten, führen Sie bitte die folgenden Schritte aus:

1. **In das Projektverzeichnis wechseln**: Navigieren Sie in das Verzeichnis src/middleware.
    ```bash
    cd src/middleware
    ```
2. **Backend Server starten**: Starten Sie den Backend Server.
    ```bash
    python app.py
    ```
