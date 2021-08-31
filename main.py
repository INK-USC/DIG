import sys, numpy as np, argparse, random
sys.path.append('../')

from tqdm import tqdm

import torch
from datasets import load_dataset
from dig import DiscretetizedIntegratedGradients
from attributions import run_dig_explanation
from metrics import eval_log_odds, eval_comprehensiveness, eval_sufficiency
import monotonic_paths

all_outputs = []


def calculate_attributions(inputs, device, args, attr_func, base_token_emb, nn_forward_func, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]

	# compute attribution
	scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd, pred	= eval_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)
	comp			= eval_comprehensiveness(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)
	suff			= eval_sufficiency(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)

	return log_odd, comp, suff


def main(args):

	# set seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# neural network specific imports
	if args.nn == 'distilbert':
		from distilbert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
	elif args.nn == 'roberta':
		from roberta_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
	elif args.nn == 'bert':
		from bert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
	else:
		raise NotImplementedError

	auxiliary_data = load_mappings(args.dataset, knn_nbrs=args.knn_nbrs)

	# Fix the gpu to use
	device		= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# init model and tokenizer in cpu first
	nn_init(device, args.dataset)

	# Define the Attribution function
	attr_func = DiscretetizedIntegratedGradients(nn_forward_func)

	# load the dataset
	if args.dataset == 'imdb':
		dataset	= load_dataset('imdb')['test']
		data	= list(zip(dataset['text'], dataset['label']))
		data	= random.sample(data, 2000)
	elif args.dataset == 'sst2':
		dataset	= load_dataset('glue', 'sst2')['test']
		data	= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
	elif args.dataset == 'rotten':
		dataset	= load_dataset('rotten_tomatoes')['test']
		data	= list(zip(dataset['text'], dataset['label']))
	else:
		raise NotImplementedError

	# get ref token embedding
	base_token_emb = get_base_token_emb(device)

	# compute the DIG attributions for all the inputs
	print('Starting attribution computation...')
	inputs = []
	log_odds, comps, suffs, count = 0, 0, 0, 0
	print_step = 2
	for row in tqdm(data):
		inp = get_inputs(row[0], device)
		input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
		scaled_features 		= monotonic_paths.scale_inputs(input_ids.squeeze().tolist(), ref_input_ids.squeeze().tolist(),\
											device, auxiliary_data, steps=args.steps, factor=args.factor, strategy=args.strategy)
		inputs					= [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask]
		log_odd, comp, suff		= calculate_attributions(inputs, device, args, attr_func, base_token_emb, nn_forward_func, get_tokens)
		log_odds	+= log_odd
		comps		+= comp
		suffs 		+= suff
		count		+= 1

		# print the metrics
		if count % print_step == 0:
			print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4))

	print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 'Sufficiency: ', np.round(suffs / count, 4))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='IG Path')
	parser.add_argument('-dataset', 	default='sst2', 		choices=['sst2', 'imdb', 'ag', 'rotten', 'sst2_epoch'])
	parser.add_argument('-nn', 			default='distilbert', 	choices=['distilbert', 'roberta', 'lstm', 'bert', 'albert'])
	parser.add_argument('-strategy', 	default='greedy', 		choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
	parser.add_argument('-steps', 		default=30, type=int)	# m
	parser.add_argument('-topk', 		default=20, type=int)	# k
	parser.add_argument('-factor', 		default=0, 	type=int)	# f
	parser.add_argument('-knn_nbrs',	default=500, type=int)	# KNN
	parser.add_argument('-seed', 		default=42, type=int)

	args = parser.parse_args()

	main(args)
