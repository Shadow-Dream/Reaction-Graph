class ConfigLoader:
    def __init__(self,default_config,config,kwargs):
        self.config = self.load_config(default_config,config,kwargs)

    def load_config_recursive(self,default_config,imcoming_config):
        if type(imcoming_config) is not dict:
            return imcoming_config
        for key,value in default_config.items():
            if key in imcoming_config:
                if type(value) is dict:
                    default_config[key] = self.load_config_recursive(value,imcoming_config[key])
                else:
                    default_config[key] = imcoming_config[key]
                del imcoming_config[key]

        for key,value in imcoming_config.items():
            default_config[key] = value
        return default_config

    def load_config(self,default_config,config,kwargs):
        config = self.load_config_recursive(config,kwargs)
        config = self.load_config_recursive(default_config,config)
        return config
    
    def apply(self,target):
        for key, value in self.config.items():
            target.__dict__[key] = value