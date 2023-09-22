import torch

def load_hf_lm_and_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        use_fast_tokenizer=False,
        padding_side="left",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map,
                                                         torch_dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings
    # to avoid wrong embedding index.
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
            model.config.max_position_embeddings))

    return model, tokenizer