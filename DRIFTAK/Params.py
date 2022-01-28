import json
import os

class Params:

    def PARAMS(self):
        return self.__params

    def __init__(self):
        with open(os.path.join(os.getcwd(), 'params.json'), "r", encoding="utf-8") as f:
            self.__params = json.load(f)

    
    def get(self, name):
        if not name in self.__params:
            raise ValueError(f'Params: parameter name {name} does not exist')
        
        return self.__params[name]
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__params, f, indent=2)