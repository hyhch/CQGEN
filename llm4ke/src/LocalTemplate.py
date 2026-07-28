from pathlib import Path

import yaml


class LocalTemplate:
    def __init__(self, body, input_vars):
        self.body = body
        self.input = input_vars

    def get(self):
        return self.body

    @classmethod
    def load(cls, input_path):
        t_config = yaml.safe_load(Path(input_path).read_text())

        body = t_config['Body']
        for k, v in t_config.get('Import', {}).items():
            body = body.replace('{%s}' % k, Path(v).read_text())

        return cls(body, t_config['Input'])
