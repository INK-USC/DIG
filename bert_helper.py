import torch, sys, pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model, tokenizer = None, None

def nn_init(device, dataset, returns=False):
	global model, tokenizer
	if dataset == 'sst2':
		tokenizer	= AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
		model		= AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2', return_dict=False)
	elif dataset == 'imdb':
		tokenizer	= AutoTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
		model		= AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb', return_dict=False)
	elif dataset == 'rotten':
		tokenizer	= AutoTokenizer.from_pretrained('textattack/bert-base-uncased-rotten-tomatoes')
		model		= AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-rotten-tomatoes', return_dict=False)

	model.to(device)
	model.eval()
	model.zero_grad()

	if returns:
		return model, tokenizer

def move_to_device(device):
	global model
	model.to(device)

def predict(model, inputs_embeds, attention_mask=None):
	return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]

def nn_forward_func(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
	global model
	embeds	= input_embed + position_embed + type_embed
	embeds	= model.bert.embeddings.dropout(model.bert.embeddings.LayerNorm(embeds))
	pred	= predict(model, embeds, attention_mask=attention_mask)
	if return_all_logits:
		return pred
	else:
		return pred.max(1).values

def load_mappings(dataset, knn_nbrs=500):
	with open(f'processed/knns/bert_{dataset}_{knn_nbrs}.pkl', 'rb') as f:
		[word_idx_map, word_features, adj] = pickle.load(f)
	word_idx_map	= dict(word_idx_map)

	return word_idx_map, word_features, adj

def construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device):
	text_ids		= tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=tokenizer.max_len_single_sentence)
	input_ids		= [cls_token_id] + text_ids + [sep_token_id]	# construct input token ids
	ref_input_ids	= [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]	# construct reference token ids

	return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)

def construct_input_ref_pos_id_pair(input_ids, device):
	global model
	seq_length			= input_ids.size(1)
	position_ids 		= model.bert.embeddings.position_ids[:,0:seq_length].to(device)
	ref_position_ids	= model.bert.embeddings.position_ids[:,0:seq_length].to(device)

	return position_ids, ref_position_ids

def construct_input_ref_token_type_pair(input_ids, device):
	seq_len				= input_ids.size(1)
	token_type_ids		= torch.tensor([[0] * seq_len], dtype=torch.long, device=device)
	ref_token_type_ids	= torch.zeros_like(token_type_ids, dtype=torch.long, device=device)
	return token_type_ids, ref_token_type_ids

def construct_attention_mask(input_ids):
	return torch.ones_like(input_ids)

def get_word_embeddings():
	global model
	return model.bert.embeddings.word_embeddings.weight

def construct_word_embedding(model, input_ids):
	return model.bert.embeddings.word_embeddings(input_ids)

def construct_position_embedding(model, position_ids):
	return model.bert.embeddings.position_embeddings(position_ids)

def construct_type_embedding(model, type_ids):
	return model.bert.embeddings.token_type_embeddings(type_ids)

def construct_sub_embedding(model, input_ids, ref_input_ids, position_ids, ref_position_ids, type_ids, ref_type_ids):
	input_embeddings				= construct_word_embedding(model, input_ids)
	ref_input_embeddings			= construct_word_embedding(model, ref_input_ids)
	input_position_embeddings		= construct_position_embedding(model, position_ids)
	ref_input_position_embeddings	= construct_position_embedding(model, ref_position_ids)
	input_type_embeddings			= construct_type_embedding(model, type_ids)
	ref_input_type_embeddings		= construct_type_embedding(model, ref_type_ids)

	return 	(input_embeddings, ref_input_embeddings), \
			(input_position_embeddings, ref_input_position_embeddings), \
			(input_type_embeddings, ref_input_type_embeddings)

def get_base_token_emb(device):
	global model
	return construct_word_embedding(model, torch.tensor([tokenizer.pad_token_id], device=device))

def get_tokens(text_ids):
	global tokenizer
	return tokenizer.convert_ids_to_tokens(text_ids.squeeze())

def get_inputs(text, device):
	global model, tokenizer
	ref_token_id = tokenizer.pad_token_id
	sep_token_id = tokenizer.sep_token_id
	cls_token_id = tokenizer.cls_token_id

	input_ids, ref_input_ids		= construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device)
	position_ids, ref_position_ids	= construct_input_ref_pos_id_pair(input_ids, device)
	type_ids, ref_type_ids			= construct_input_ref_token_type_pair(input_ids, device)
	attention_mask					= construct_attention_mask(input_ids)

	(input_embed, ref_input_embed), (position_embed, ref_position_embed), (type_embed, ref_type_embed) = \
				construct_sub_embedding(model, input_ids, ref_input_ids, position_ids, ref_position_ids, type_ids, ref_type_ids)

	return [input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask]
