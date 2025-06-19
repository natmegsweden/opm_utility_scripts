import yaml


def get_config(config_file: str):
    if config_file:
        all_config = yaml.safe_load(open(config_file, 'r'))
    
    project = all_config.get('project', {})
    opm = all_config.get('opm', {})
    
    config = {
        'tasks': project.get('tasks', []),
        'squidMEG': project.get('squidMEG', {}),
        'opmMEG': project.get('opmMEG', {}),
        'hpi_file': opm.get('hpi_file', ''),
        'hpi_freq': opm.get('hpi_freq', 33.0),
        'downsample_freq': opm.get('downsample_to_hz', 1000),
        'plot': opm.get('plot', False)
    }
    return config