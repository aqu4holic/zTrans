from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
import torch
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from transformers.utils import logging
logging.set_verbosity_error()
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

seed = 18022004
np.random.seed(seed)
set_seed(seed)

data_prefix: str = 'data'
repo_prefix: str = f'{data_prefix}/repos'

data_name = 'data_method_30k_test.parquet'

model_id: str = 'deepseek-ai/deepseek-coder-6.7b-instruct'

prompt_template: str = '''rewrite below method from library "{}" to "{}". ONLY WRITE CODE WITH NO COMMENTS, IMPORTS, TEXT.
```
{}
```
'''

def build_prompts(data_df: pd.DataFrame) -> List[Any]:
    prompts: List[Any] = []
    shortlisted_prompt: List[Any] = []

    BEGIN_TOKEN: str = '<｜fim▁begin｜>'
    FILL_TOKEN: str = '<｜fim▁hole｜>'
    END_TOKEN: str = '<｜fim▁end｜>'

    for id in tqdm(range(len(data_df)), desc = 'Building prompts'):
        line = data_df.iloc[id]

        from_lib: str = line['fromLib']
        to_lib: str = line['toLib']
        method_before: str = line['methods_before']
        method_after: str = line['methods_after']
        file_name = line['fileName']

        if (len(method_before) == 0 or len(method_after) == 0):
            continue

        prompt: str = prompt_template.format(from_lib, to_lib, method_before)
        ground_truth: str = line['methods_after']

        prompts.append({'id': line['id'], 'prompt': prompt, 'ground_truth': ground_truth})

    return prompts

def build_inputs(prompts: List[Any], tokenizer: AutoTokenizer) -> List[Any]:
    messages: List[Any] = []
    valid_inputs: List[Any] = []

    for id in tqdm(range(len(prompts)), desc = 'Building inputs'):
        sample = prompts[id]

        messages = [(
            {
                'role': 'user',
                'content': sample['prompt'],
            }
        )]

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors = 'pt').to('cpu')

        if (len(inputs[0]) > 256):
            continue

        valid_inputs.append({
            'id': sample['id'],
            'inputs': inputs,
            'prompt': sample['prompt'],
        })

    print('-' * 50)
    print(f'valid inputs: {len(valid_inputs)}')
    print('-' * 50)
    print()

    return valid_inputs

def build_tokenizer(model_id: str = model_id) -> AutoTokenizer:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True,)

    return tokenizer

def build_model(model_id: str = model_id) -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit = True,
    )

    # device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,
        quantization_config = quantization_config,
        torch_dtype = torch.float16,
        device_map = 'auto',
    )

    return model


def decode_outputs(tokenizer: AutoTokenizer, inputs: List[Any], outputs: List[Any]) -> List[Any]:
    results: List[Any] = []

    with torch.no_grad():
        for output in tqdm(outputs, desc = 'Decoding'):
            id = output['id']
            single_inputs = output['inputs']
            single_outputs = output['outputs']
            prompt = output['prompt']

            decoded_output = tokenizer.decode(single_outputs[0][len(single_inputs[0]):], skip_special_tokens = True)

            results.append(
                {
                    'id': id,
                    'output': decoded_output,
                    'prompt': prompt,
                }
            )

    return results

def save_results(results: List[str], data_df: pd.DataFrame):
    valid_ids = [result['id'] for result in results]
    res_df = data_df[data_df['id'].isin(valid_ids)].copy()

    res_df['predicted'] = ''
    res_df['prompt'] = ''

    # print(res_df.head())

    for id in range(len(results)):
        sample = results[id]

        res_df.loc[res_df['id'] == sample['id'], 'prompt'] = sample['prompt']
        res_df.loc[res_df['id'] == sample['id'], 'predicted'] = sample['output']

    res_df.to_parquet(f'{data_prefix}/results_{data_name}', engine = 'pyarrow')

def generate_from_inputs(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, valid_inputs: List[Any], data_df: pd.DataFrame) -> List[Any]:
    outputs: List[Any] = []

    for sample in tqdm(valid_inputs, desc = 'Generating'):
        id = sample['id']
        single_inputs = sample['inputs']
        prompt = sample['prompt']

        single_inputs = single_inputs.to(model.device)
        single_outputs = model.generate(
            single_inputs,
            max_new_tokens = 256,
            do_sample = False,
            top_k = 50,
            top_p = 0.95,
            eos_token_id = tokenizer.eos_token_id,
        )

        single_inputs = single_inputs.to('cpu')

        outputs.append(
            {
                'id': id,
                'inputs': single_inputs,
                'outputs': single_outputs,
                'prompt': prompt,
            }
        )

        # save results every 15 samples
        if (len(outputs) % 15 == 0):
            results: List[Any] = decode_outputs(tokenizer = tokenizer, inputs = valid_inputs, outputs = outputs)

            save_results(results = results, data_df = data_df)

            print(f'saved results for {len(outputs)} samples')
            print('-' * 50)
            print()

    return outputs

def main():
    data_df: pd.DataFrame = pd.read_parquet(f'{data_prefix}/{data_name}', engine = 'pyarrow')

    tokenizer: AutoTokenizer = build_tokenizer(model_id = model_id)

    prompts: List[Any] = build_prompts(data_df = data_df)
    valid_inputs = build_inputs(prompts = prompts, tokenizer = tokenizer)

    model: AutoModelForCausalLM = build_model(model_id = model_id)
    outputs: List[Any] = generate_from_inputs(model = model, tokenizer = tokenizer, valid_inputs = valid_inputs, data_df = data_df)
    results: List[Any] = decode_outputs(tokenizer = tokenizer, inputs = valid_inputs, outputs = outputs)

    save_results(results = results, data_df = data_df)

    print('Done!')

if (__name__ == '__main__'):
    main()