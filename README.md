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