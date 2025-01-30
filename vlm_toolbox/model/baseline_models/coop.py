
import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch import nn

from model.baseline_models.soft_prompt_vlm import SoftPromptVLM
from model.clip_modeling.openai import clip


class PromptLearner(nn.Module):
    def __init__(self, config, labels, token_embedding, hidden_dim, tokenizer, device='cpu', label_id_prompt_id_mapping=None):
        super().__init__()
        if label_id_prompt_id_mapping is None:
             label_id_prompt_id_mapping = torch.zeros(len(labels)).to(device)
        
        self.register_buffer('label_id_prompt_id_mapping', label_id_prompt_id_mapping.int().to(device))
        n_cls = len(torch.unique(self.label_id_prompt_id_mapping))
        
        n_ctx = config.get_text_n_context()
        ctx_init = config.get_text_context_initialization()
        ctx_dim = hidden_dim

        if ctx_init is not None:
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = token_embedding(prompt)

            n_ctx = len(ctx_init.split(" "))
            ctx_vectors = embedding[0, 1:1 + n_ctx, :].unsqueeze(0).expand(n_cls, -1, -1)
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors).to(device)

        prompts = [prompt_prefix + " " + label +  "." for label in labels]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = config.get_class_token_position()
        self.ctx_init = ctx_init

    def forward(self, label_ids):
        context_ids = self.label_id_prompt_id_mapping[label_ids]
        ctx = self.ctx[context_ids, :, :]
        prefix = self.token_prefix[label_ids]
        suffix = self.token_suffix[label_ids]

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise NotImplementedError("Class token positioning other than 'end' is not implemented.")
    
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, prompt_learner, clip_model):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.token_embedding = clip_model.token_embedding

    def encode_text(self, label_ids, **kwargs):
        prompts = self.prompt_learner(label_ids)
        tokenized_prompts = self.prompt_learner.tokenized_prompts[label_ids]
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features

    def encode_image(self, images, **kwargs):
        return self.image_encoder(images)

class COOP(SoftPromptVLM):
    @staticmethod
    def from_pretrained(model_config, **kwargs):
        soft_prompting_config = model_config['soft_prompting_config']

        labels = model_config['labels']
        label_id_prompt_id_mapping = model_config['label_id_prompt_id_mapping']
        
        clip_model, _ = COOP.build_model(
            model_config['model_url'],
            'coop',
            soft_prompting_config,
        )
        tokenizer = _Tokenizer()
        device = next(iter(clip_model.parameters())).device

        prompt_learner = PromptLearner(
            soft_prompting_config,
            labels,
            clip_model.token_embedding,
            clip_model.token_embedding.weight.shape[1],
            tokenizer,
            device=device,
            label_id_prompt_id_mapping=label_id_prompt_id_mapping,
        )

        model = CustomCLIP(prompt_learner, clip_model)

        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        return COOP(
            model,
            model_config=model_config,
            soft_prompting_config=soft_prompting_config,
            **kwargs,
        )

    def register_new_labels(self, labels, label_id_prompt_id_mapping=None, use_learned_contex=True):
        device = self.device

        if label_id_prompt_id_mapping is None:
            assert self.model.prompt_learner.n_cls == 1

        new_prompt_learner = PromptLearner(
            self.soft_prompting_config,
            labels,
            self.model.token_embedding,
            self.model.token_embedding.weight.shape[1],
            _Tokenizer(),
            device=device,
            label_id_prompt_id_mapping=label_id_prompt_id_mapping.to(device),
        ).to(device=device)

        if use_learned_contex:
            assert len(torch.unique(self.model.prompt_learner.label_id_prompt_id_mapping)) == 1
            assert len(torch.unique(new_prompt_learner.label_id_prompt_id_mapping)) == 1
            
            ctx_vectors = self.model.prompt_learner.ctx[0].unsqueeze(0).expand(new_prompt_learner.n_cls, -1, -1)
            new_prompt_learner.ctx = nn.Parameter(ctx_vectors).to(dtype=self.dtype)

        self.model.prompt_learner = new_prompt_learner


    def state_dict(self, destination=None, prefix='prompt_learner.', keep_vars=False):
        state = {}
        state[prefix + 'ctx'] = self.model.prompt_learner.ctx.data
        state[prefix + 'token_prefix'] = self.model.prompt_learner.token_prefix.data
        state[prefix + 'token_suffix'] = self.model.prompt_learner.token_suffix.data
        return state


    def load_state_dict(self, state_dict, strict=False, assign=False, logging_fn=print):
        key_to_replace = 'prompt_learner.ctx'
        if key_to_replace in state_dict.keys():
            ctx = state_dict[key_to_replace]
            if len(ctx.shape) == 2:
                ctx = ctx.unsqueeze(0)

            module = self.model.prompt_learner.ctx
            if ctx.shape != module.shape:
                logging_fn(f"Replacing {key_to_replace} parameters (curr. shape: {ctx.shape}), prev. shape: {module.shape}")
                self.model.prompt_learner.ctx = nn.Parameter(ctx).to(device=module.device, dtype=module.dtype)
                self.model.prompt_learner.n_ctx = ctx.shape[1]
                del state_dict[key_to_replace]

        return super().load_state_dict(state_dict, strict=False, assign=assign, logging_fn=logging_fn)
            
    def show(self, logging_fn=print):
        super().show(logging_fn=logging_fn)
        logging_fn(f'\nLearnable context vectors length = {self.model.prompt_learner.n_ctx}')
        logging_fn(f'Learnable context vectors cnt = {self.model.prompt_learner.n_cls}')
        logging_fn(f'Context initialization = {self.model.prompt_learner.ctx_init}')
        logging_fn(f'Class token position in contexts = {self.model.prompt_learner.class_token_position}\n')
        return self
        














