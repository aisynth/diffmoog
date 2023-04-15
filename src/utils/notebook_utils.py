import ipywidgets as widgets

from collections import defaultdict
import re

import numpy as np

from synth.synth_constants import SynthConstants


def create_synth_params_layout(synth_preset):

    synth_structure = SynthConstants()

    layouts = []
    output = {}
    for cell in synth_preset:

        index = cell['index']
        operation = cell['operation']

        cell_params = synth_structure.modular_synth_params[operation]
        sampling_config = synth_structure.param_configs[operation]

        if operation == 'env_adsr':
            adsr_widgets = {f'{index}-{operation}-{p}':
                            widgets.FloatSlider(value=0.33, min=0.0, max=1.0, description=p) for p in sampling_config}
            layouts.extend([widgets.Label(f"{index} ADSR"), widgets.HBox(list(adsr_widgets.values()))])
            output.update(adsr_widgets)
            continue

        cell_widgets = []
        for param_name, param_config in sampling_config.items():
            if param_config['type'] == 'freq_c':
                widget = widgets.FloatSlider(value=synth_structure.middle_c_freq, min=synth_structure.osc_freq_list[0],
                                             max=synth_structure.osc_freq_list[-1], description=param_name)
            elif param_config['type'] == 'uniform':
                widget = widgets.FloatSlider(value=np.mean(param_config['values']), min=param_config['values'][0],
                                             max=param_config['values'][1], description=param_name)
            elif param_config['type'] == 'choice':
                widget = widgets.Dropdown(options=param_config['values'], description=param_name)
            else:
                continue

            cell_widgets.append(widget)
            output[f'{index}-{operation}-{param_name}'] = widget

        layouts.extend([widgets.Label(f"{index} {operation.title()}"), widgets.HBox(cell_widgets)])

    gen_button = widgets.Button(description="Generate")
    output_window = widgets.Output()

    full_layout = widgets.VBox(layouts + [gen_button, output_window])

    return full_layout, output, gen_button, output_window


def convert_flat_nb_dict_to_synth(flat_dict):

    synth_structure = SynthConstants()

    index_re = re.compile("\((\d), (\d)\)")

    parsed_dict = defaultdict(dict)
    for k, v in flat_dict.items():
        cell_index_str = k.split('-')[0]
        cell_index = tuple([int(x) for x in re.match(index_re, cell_index_str).groups()])

        operation = k.split('-')[1]
        param_name = k.split('-')[-1]

        expected_params = synth_structure.modular_synth_params[operation]

        if cell_index in parsed_dict:
            parsed_dict[cell_index]['parameters'][param_name] = v
        else:
            parsed_dict[cell_index]['parameters'] = {param_name: v}

        parsed_dict[cell_index]['operation'] = operation

        if 'active' in expected_params:
            parsed_dict[cell_index]['parameters']['active'] = [True]
        if 'fm_active' in expected_params:
            parsed_dict[cell_index]['parameters']['fm_active'] = [True]

    return parsed_dict


