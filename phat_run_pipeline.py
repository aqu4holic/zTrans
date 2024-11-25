from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
import datasets
import torch
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
import argparse
from tqdm import tqdm
import time

from transformers.utils import logging
logging.set_verbosity_error()
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

seed = 18022004
np.random.seed(seed)
set_seed(seed)

data_prefix: str = 'data'
repo_prefix: str = f'{data_prefix}/repos'

prompt_template: str = '''rewrite below method from library "{}" to "{}". ONLY WRITE CODE WITH NO COMMENTS, IMPORTS, TEXT.
```
{}
```
'''

batch_prompt_template: str = '''<｜begin▁of▁sentence｜>### Instruction:
you're a software engineer working on a project. ONLY RESPOND WITH CODE, NO COMMENTS, IMPORTS, TEXT, NO EXPLAIN.
rewrite below method from library "{}" to "{}". ONLY WRITE THE METHOD BODY IN ```CODE```.
```
{}
```

### Response:
'''

def calculate_time(function: Callable, *args, **kwargs) -> Any:
    start_time: float = time.time()
    result = function(*args, **kwargs)
    end_time: float = time.time()
    print(f'Executed {function.__name__} in {end_time - start_time} seconds')
    print('-' * 50)

    return result

def build_prompts(data_df: pd.DataFrame, batched: bool) -> List[Any]:
    prompts: List[Any] = []

    BEGIN_TOKEN: str = '<｜fim▁begin｜>'
    FILL_TOKEN: str = '<｜fim▁hole｜>'
    END_TOKEN: str = '<｜fim▁end｜>'

    for id in tqdm(range(len(data_df)), desc = 'Building prompts'):
        line = data_df.iloc[id]

        from_lib: str = line['fromLib']
        to_lib: str = line['toLib']
        method_before: str = line['method_before']
        ground_truth: str = line['method_after']

        if (batched):
            prompt: str = batch_prompt_template.format(from_lib, to_lib, method_before)
        else:
            prompt: str = prompt_template.format(from_lib, to_lib, method_before)
        ground_truth: str = line['method_after']

        prompts.append({'id': line['id'], 'prompt': prompt, 'ground_truth': ground_truth})

    return prompts

def build_prompts_hf(data_df: datasets.arrow_dataset.Dataset) -> List[Any]:
    prompts: List[Any] = []

    BEGIN_TOKEN: str = '<｜fim▁begin｜>'
    FILL_TOKEN: str = '<｜fim▁hole｜>'
    END_TOKEN: str = '<｜fim▁end｜>'

    for id in tqdm(range(len(data_df)), desc = 'Building prompts'):
        line = data_df[id]

        from_lib: str = line['fromLib']
        to_lib: str = line['toLib']
        method_before: str = line['method_before']
        ground_truth: str = line['method_after']

        prompt: str = prompt_template.format(from_lib, to_lib, method_before)

        prompts.append({'id': line['id'], 'prompt': prompt, 'ground_truth': ground_truth})

    return prompts

def build_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    model_id: str = args.model
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True,)

    return tokenizer

def build_model(args: argparse.Namespace) -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit = True,
    )

    device_id: str = args.device
    model_id: str = args.model
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    # device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,
        quantization_config = quantization_config,
        torch_dtype = torch.float16,
        device_map = 'auto',
    )

    return model

def decode_outputs(tokenizer: AutoTokenizer, outputs: List[Any]) -> List[Any]:
    results: List[Any] = []

    with torch.no_grad():
        for output in tqdm(outputs, desc = 'Decoding'):
            id = output['id']
            single_inputs = output['inputs']
            single_outputs = output['outputs']
            prompt = output['prompt']

            decoded_output = tokenizer.decode(single_outputs[len(single_inputs):], skip_special_tokens = True)

            results.append(
                {
                    'id': id,
                    'output': decoded_output,
                    'prompt': prompt,
                }
            )

    return results

def save_results(args: argparse.Namespace, results: List[str], data_df: pd.DataFrame):
    output_name: str = args.output_file

    valid_ids = [result['id'] for result in results]
    res_df = data_df[data_df['id'].isin(valid_ids)].copy()

    res_df['predicted'] = ''
    res_df['prompt'] = ''

    for id in range(len(results)):
        sample = results[id]

        res_df.loc[res_df['id'] == sample['id'], 'prompt'] = sample['prompt']
        res_df.loc[res_df['id'] == sample['id'], 'predicted'] = sample['output']

    res_df.to_parquet(f'{data_prefix}/{output_name}', engine = 'pyarrow')

def build_message_inputs(prompts: List[Any], tokenizer: AutoTokenizer) -> List[Any]:
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

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt = True, padding = True, truncation = True, return_tensors = 'pt').to('cpu')

        valid_inputs.append({
            'id': sample['id'],
            'inputs': inputs,
            'prompt': sample['prompt'],
        })

    return valid_inputs

def generate_from_inputs(args: argparse.Namespace, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, valid_inputs: List[Any], data_df: pd.DataFrame) -> List[Any]:
    outputs: List[Any] = []

    max_new_tokens: int = args.max_new_tokens
    do_sample: bool = args.do_sample
    top_k: int = args.top_k
    top_p: float = args.top_p

    for sample in tqdm(valid_inputs, desc = 'Generating'):
        id = sample['id']
        single_inputs = sample['inputs']
        prompt = sample['prompt']

        single_inputs = single_inputs.to(model.device)
        single_outputs = model.generate(
            single_inputs,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            top_k = top_k,
            top_p = top_p,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
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
            results: List[Any] = decode_outputs(tokenizer = tokenizer, outputs = outputs)

            save_results(args = args, results = results, data_df = data_df)

            print(f'saved results for {len(outputs)} samples')
            print('-' * 50)
            print()

    return outputs

def build_batched_inputs(args: argparse.Namespace, prompts: List[Any], tokenizer: AutoTokenizer) -> List[Any]:
    inputs: List[Any] = tokenizer.batch_encode_plus(prompts, padding = True, truncation = True, return_tensors = 'pt').to('cpu')

    return inputs

def generate_from_prompts(args: argparse.Namespace, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: List[Any], data_df: pd.DataFrame) -> List[Any]:
    outputs: List[Any] = []

    batch_size: int = args.batch_size
    max_new_tokens: int = args.max_new_tokens
    do_sample: bool = args.do_sample
    top_k: int = args.top_k
    top_p: float = args.top_p

    def create_batches(prompts: List[Any], batch_size: int) -> List[List[Any]]:
        batches: List[List[Any]] = []

        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i:i + batch_size])

        return batches

    batches: List[List[Any]] = create_batches(prompts, batch_size)

    for batch_id, batch in tqdm(enumerate(batches), desc = 'Generating by batches'):
        # valid_inputs: List[Any] = build_inputs(batch, tokenizer)

        # print(len(valid_inputs))
        prompts = [sample['prompt'] for sample in batch]

        inputs: List[Any] = build_batched_inputs(args = args, prompts = prompts, tokenizer = tokenizer)
        inputs = inputs.to(model.device)
        batch_outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            top_k = top_k,
            top_p = top_p,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
        )
        inputs = inputs.to('cpu')
        torch.cuda.empty_cache()

        formatted_outputs = [
            {
                'id': batch[id]['id'],
                'inputs': inputs[id],
                'outputs': val,
                'prompt': batch[id]['prompt'],
            } for id, val in enumerate(batch_outputs)
        ]

        outputs.extend(formatted_outputs)

        # with torch.no_grad():
        #     truncated_outputs = [val[len(inputs[id]):] for id, val in enumerate(batch_outputs)]
        #     batch_decoded_outputs = tokenizer.batch_decode(truncated_outputs, skip_special_tokens = True)

        #     for i in range(len(batch)):
        #         outputs.append(
        #             {
        #                 'output': batch_decoded_outputs[i],
        #                 'id': batch[i]['id'],
        #                 'prompt': batch[i]['prompt'],
        #             }
        #         )

        # save results every 5 batches
        if (batch_id % 5 == 0):
            results: List[Any] = decode_outputs(tokenizer = tokenizer, outputs = outputs)

            save_results(args = args, results = results, data_df = data_df)

            print(f'saved results for {len(outputs)} samples')
            print('-' * 50)
            print()

    return outputs

def run(args: argparse.Namespace):
    start_time: float = time.time()
    data_name: str = args.input_file
    datatset_id: str = args.dataset_id
    split: str = args.split
    data_df: pd.DataFrame = datasets.load_dataset(datatset_id, split = split).to_pandas()
    end_time: float = time.time()
    print(f'Loaded data in {end_time - start_time} seconds')
    print('-' * 50)

    # start_time = time.time()
    # tokenizer: AutoTokenizer = build_tokenizer(args = args)
    # end_time = time.time()
    # print(f'Built tokenizer in {end_time - start_time} seconds')
    # print('-' * 50)
    tokenizer: AutoTokenizer = calculate_time(build_tokenizer, args = args)

    # start_time = time.time()
    # prompts: List[Any] = build_prompts(data_df = data_df)
    # valid_inputs: List[Any] = build_inputs(prompts = prompts, tokenizer = tokenizer)
    # end_time = time.time()
    # print(f'Built prompts in {end_time - start_time} seconds')
    prompts: List[Any] = calculate_time(build_prompts, data_df = data_df, batched = True)
    print(f'Valid prompts: {len(prompts)}')
    print('-' * 50)

    # start_time = time.time()
    # model: AutoModelForCausalLM = build_model(args = args)
    # end_time = time.time()
    # print(f'Built model in {end_time - start_time} seconds')
    # print('-' * 50)
    model: AutoModelForCausalLM = calculate_time(build_model, args = args)

    # start_time = time.time()
    # outputs: List[Any] = generate_from_inputs(args = args, model = model, tokenizer = tokenizer, valid_inputs = valid_inputs, data_df = data_df)
    # results: List[Any] = decode_outputs(tokenizer = tokenizer, inputs = valid_inputs, outputs = outputs)
    # end_time = time.time()
    # print(f'Generated in {end_time - start_time} seconds')
    # print('-' * 50)
    # outputs: List[Any] = calculate_time(generate_from_inputs, args = args, model = model, tokenizer = tokenizer, valid_inputs = valid_inputs, data_df = data_df)
    outputs: List[Any] = calculate_time(generate_from_prompts, args = args, model = model, tokenizer = tokenizer, prompts = prompts, data_df = data_df)
    results: List[Any] = calculate_time(decode_outputs, tokenizer = tokenizer, outputs = outputs)

    # start_time = time.time()
    # save_results(args = args, results = results, data_df = data_df)
    # end_time = time.time()
    # print(f'Saved results to {args.output_file} in {end_time - start_time} seconds')
    # print('-' * 50)
    calculate_time(save_results, args = args, results = results, data_df = data_df)

    print('Done!')

def main():
    parser = argparse.ArgumentParser(description = 'Process a file.')

    # data parameters
    parser.add_argument('--input_file', type = str, nargs = '?', default = 'sampled_no_code.parquet', help = 'The name of the file to process')
    parser.add_argument('--output_file', type = str, nargs = '?', default = 'sampled_code.parquet', help = 'The name of the file to output')
    parser.add_argument('--dataset_id', type = str, nargs = '?', default = 'blackwhite1337/zTrans_dataset', help = 'Dataset ID on Huggingface')
    parser.add_argument('--split', type = str, nargs = '?', default = 'test', help = 'Dataset split to use')

    # model parameters
    parser.add_argument('--model', type = str, nargs = '?', default = 'deepseek-ai/deepseek-coder-6.7b-instruct', help = 'Model ID on Huggingface')
    parser.add_argument('--device', nargs = '?', default = '0', help = 'GPU ID to use')
    parser.add_argument('--batch_size', type = int, nargs = '?', default = 2, help = 'Batch size per CPU/GPU for generation')

    # generation parameters
    parser.add_argument('--max_length', type = int, nargs = '?', default = 256, help = 'Max length of the prompt')
    parser.add_argument('--max_new_tokens', type = int, nargs = '?', default = 256, help = 'Max new tokens to generate')
    parser.add_argument('--do_sample', type = bool, nargs = '?', default = False, help = 'Whether to sample or not')
    parser.add_argument('--top_k', type = int, nargs = '?', default = 50, help = 'Top k tokens to sample from')
    parser.add_argument('--top_p', type = float, nargs = '?', default = 0.95, help = 'Top p tokens to sample from')

    args = parser.parse_args()

    run(args = args)

if (__name__ == '__main__'):
    main()