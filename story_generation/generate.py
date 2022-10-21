import torch
import sys
from simctg import SimCTG
import pickle 
import os
import pandas as pd
from tqdm import tqdm
import fire
import os
import shutil

def do_evaluation(gen_path, experiment_name):
    abs_gen_path = os.path.abspath(gen_path)
    os.system(f'cd ../../TextGenMetrics; source /opt/conda/etc/profile.d/conda.sh; conda activate eval; CUDA_VISIBLE_DEVICES=7 nohup python evaluate_metrics.py --gen_path {abs_gen_path} --format simctg --out_path {experiment_name}.tsv &> {experiment_name}.out &')


def main(model_path = r'/workspace/SimCTG/story_generation/simctg_cnndm_aug_min3_bsz16/training_step_30000_train_mle_loss_2.897_train_cl_loss_0.002_dev_ppl_17.362', 
            gen_dir = 'generated_texts/simctg_cnndm/contrastive_search_fixed',
            prefix = ''):
    # model_path = r'/workspace/SimCTG/story_generation/simctg_cnndm_aug_min3_bsz16/training_step_30000_train_mle_loss_2.897_train_cl_loss_0.002_dev_ppl_17.362'
    # gen_dir = f'generated_texts/simctg_cnndm/contrastive_search_fixed'
    
    experiment_name = model_path.split("/")[3]
    gen_path = f'{gen_dir}/gen.txt'

    pad_token = '<_PAD_>'
    model = SimCTG(model_path, pad_token).cuda()
    model.eval()

    test_examples = pickle.load(open(f'../../progressive-generation/data/cnn/test.pickle', 'rb'))
    os.makedirs(gen_dir, exist_ok=True)

    log_file = open(f'{gen_dir}/log.txt', 'w')
    gens = []
    for idx, example in enumerate(tqdm(test_examples, desc='Generating')):
        # if idx > 2: break
        # condition = r'''A bomb blast at a coffee shop north of Baghdad kills one, wounds nine, police say. An anti-government demonstration organizer is gunned down in Haditha, police say. Roadside bomb blasts in Baghdad killed three, wound six, police say.'''
        # if idx > 3 : break
        condition, truth = example['condition'], example['text']
        if prefix : # add prefix if there's any
            condition = f"{prefix} {condition}"
        tokens = model.tokenizer.tokenize(condition)
        input_ids = model.tokenizer.convert_tokens_to_ids(tokens)   
        input_ids = torch.LongTensor(input_ids).view(1,-1).cuda()

        beam_width, alpha, decoding_len = 5, 0.65, 1020 - input_ids.size(-1)
        with torch.no_grad():
            whole_out, output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
        generated_story = model.tokenizer.decode(output)
        # generated_story = generated_story.replace(condition, "")
        generated_story = generated_story.replace(pad_token, " ").replace(model.tokenizer.eos_token, "").strip()
        # generated_story = generated_story.split(pad_token)[0] # truncate in case generated stories are too long

        gens.append((condition, truth, generated_story))
    result_df = pd.DataFrame.from_records(gens, columns = ["CONDITION", "TRUTH", "GENERATED"])
    result_df.to_csv(gen_dir + "/gen.txt" ,index=False)

    # do evaluation
    do_evaluation(gen_path = gen_path, experiment_name = experiment_name)

# load model
if __name__ == "__main__":
    fire.Fire(main)

