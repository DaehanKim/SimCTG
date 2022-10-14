import torch
import sys
from simctg import SimCTG
import pickle 
import os
import pandas as pd
from tqdm import tqdm

# load model
model_path = r'/workspace/SimCTG/story_generation/simctg_cnndm/training_step_29000_train_mle_loss_2.756_train_cl_loss_0.002_dev_ppl_16.614'
pad_token = '<_PAD_>'
model = SimCTG(model_path, pad_token).cuda()
model.eval()

test_examples = pickle.load(open(f'../../progressive-generation/data/cnn/test.pickle', 'rb'))
gen_dir = f'generated_texts/simctg_cnndm/contrastive_search'
os.makedirs(gen_dir, exist_ok=True)

log_file = open(f'{gen_dir}/gen.txt', 'w')
gens = []
for idx, example in enumerate(tqdm(test_examples, desc='Generating')):
    # condition = r'''A bomb blast at a coffee shop north of Baghdad kills one, wounds nine, police say. An anti-government demonstration organizer is gunned down in Haditha, police say. Roadside bomb blasts in Baghdad killed three, wound six, police say.'''
    # if idx > 3 : break
    condition, truth = example['condition'], example['text']

    tokens = model.tokenizer.tokenize(condition)
    input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.LongTensor(input_ids).view(1,-1).cuda()

    beam_width, alpha, decoding_len = 5, 0.65, 1020 - input_ids.size(-1)
    with torch.no_grad():
        output = model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len)
    try:
        generated_story = model.tokenizer.decode(output).split(model.tokenizer.eos_token)[1].strip()
    except:
        generated_story = model.tokenizer.decode(output)
    gens.append((condition, truth, generated_story))
result_df = pd.DataFrame.from_records(gens, columns = ["CONDITION", "TRUTH", "GENERATED"])
result_df.to_csv(gen_dir + "/gen.txt" ,index=False)