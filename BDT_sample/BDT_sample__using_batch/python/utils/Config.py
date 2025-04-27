#!/usr/bin/env python

class Config:
    def __init__(self, value:dict = {}):
        self.data = value.copy()

    def __str__(self):
        return str(self.data)

    def splitPath(self, path:str):
        keys = []
        paths = []

        for p in path.strip().split('/'):
            if p == '': continue
            keys.append(p)

        if len(keys) > 0:
            paths.append(keys[0])
            p = keys[0]
            for key in keys[1:]:
                p = f'{p}/{key}'
                paths.append(p)
        
        return keys, paths

    def __getitem__(self, key:str):
        keys, paths = self.splitPath(key)
        obj = self.data
        for key in keys:
            if key not in obj: return None
            obj = obj[key]
        return obj

    def __setitem__(self, key:str, value):
        key = key.strip('/')
        keys, paths = self.splitPath(key)
        if len(keys) == 0:
            self.data = value
            return

        obj = self.data
        ## Walk through the data tree, set create new node if not found
        for ikey, ipath in list(zip(keys, paths))[:-1]:
            if ikey not in obj:
                obj[ikey] = {}
            elif type(obj[ikey]) != type({}):
                print(f"@@@ end-node {ipath} already exists. Replacing the node...")
                obj[ikey] = {}
            obj = obj[ikey]

        if keys[-1] in obj:
            print(f"@@@ node {key} already exists. Replacing the node...")

        obj[keys[-1]] = value

    def __iadd__(self, value:dict): ## += operator, to override existing config
        ## find all end-nodes of input values
        endnodes = self.findEndnodes(value, '/')
        value = Config(value)
        for path in endnodes:
            self.__setitem__(path, value[path])
        return self

    def findEndnodes(self, value:dict, prefix:str):
        endnodes = []
        for key, val in value.items():
            path = f'{prefix}/{key}' if prefix != '' else key
            if type(val) == dict:
                endnodes.extend(self.findEndnodes(val, path))
            else:
                endnodes.append(path)
        return endnodes

def overrideConfig(config, key, obj, attr, astype=None):
    value = getattr(obj, attr) or config[key]
    if astype: value = astype(value)
    config[key] = value
    setattr(obj, attr, value)

if __name__ == '__main__':
    x = Config({'a':1, 'b':{'bc':2,},})
    print(x)
