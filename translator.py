from datasets import load_dataset
from IPython.display import display
from IPython.html import widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm_notebook
import pickle

class Translator():

    def __init__(self, size: str, cuda: bool):
        self.model_repo = 'google/mt5-{}'.format(size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_repo)
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()
        self.max_seq_len = self.model.config.max_length #20 in this case
        self.LANG_TOKEN_MAPPING = {'en': '<en>',
                                   'de': '<de>'}
        self.special_tokens_dict = {'additional_special_tokens': list(self.LANG_TOKEN_MAPPING.values())}
        self.tokenizer.add_special_tokens(self.special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def encode_input_str(self, text, traget_lang, seq_len):
        traget_lang_token = self.LANG_TOKEN_MAPPING[traget_lang]

        # Tokenize and add pecial tokens
        input_ids = self.tokenizer.encode(text = traget_lang_token + text,
                                          return_tensors = 'pt',
                                          padding = 'max_length',
                                          truncation = True,
                                          max_length = seq_len)

        return input_ids[0]

    def encode_target_str(self, text, seq_len):
        token_ids = self.tokenizer.encode(text = text, 
                                          return_tensors = 'pt',
                                          padding = 'max_length',
                                          truncation = True,
                                          max_length = seq_len)

        return token_ids[0]

    def format_translation_data(self, translations, seq_len=128):
      
      # Choose 2 random languages for input and output
      langs = list(self.LANG_TOKEN_MAPPING.keys())
      input_lang, target_lang = np.random.choice(langs, size=2, replace=False)
      
      # Get the translations for the batch
      input_text = translations[input_lang]
      target_text = translations[target_lang]

      if input_text is None or target_text is None:
        return None

      input_token_ids = self.encode_input_str(input_text, 
                                              target_lang,
                                              seq_len)
      
      target_token_ids = self.encode_target_str(target_text,
                                                seq_len)
      
      return input_token_ids, target_token_ids

    def transform_batch(self, batch):
      inputs=[]
      targets=[]
      for translation_set in batch['translation']:
        formatted_data = self.format_translation_data(translation_set,
                                                      self.max_seq_len)
        
        if formatted_data is None:
          continue

        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))

      if self.cuda:
        batch_input_ids = torch.cat(inputs).cuda()
        batch_target_ids = torch.cat(targets).cuda()
      else:
        batch_input_ids = torch.cat(inputs)
        batch_target_ids = torch.cat(targets)

      return batch_input_ids, batch_target_ids

    def get_data_generator(self, dataset, batch_size = 32):
      dataset = dataset.shuffle()
      for i in range(0, len(dataset), batch_size):
        raw_batch = dataset[i:i+batch_size]
        yield self.transform_batch(raw_batch)

    def eval_model(model, gdataset, max_iters=8):
      test_generator = get_data_generator(gdataset, LANG_TOKEN_MAPPING,
                                          tokenizer, batch_size)
      eval_losses = []
      for i, (input_batch, label_batch) in enumerate(test_generator):
        if i >= max_iters:
          break

        model_out = model.forward(
            input_ids = input_batch,
            labels = label_batch)
        eval_losses.append(model_out.loss.item())

      return np.mean(eval_losses)

    def fit(self, data, nr_epochs = 5, batch_size = 16, lr = 5e-4, optimizer = 'AdamW', nr_batches = 1, total_steps = 1, nr_warm_steps = 1, scheduler = True, print_freq = 50, save_path: str = None, losses_path: str = None, checkpoint_freq: int = 1000):
        if optimizer == 'AdamW':
          optimizer = AdamW(self.model.parameters(), lr = lr)

        if nr_batches == 1:
          nr_batches = int(np.ceil(len(data) / batch_size))

        if total_steps == 1:
          total_steps = nr_epochs * nr_batches
        
        if nr_warm_steps == 1:
          nr_warm_steps = int(total_steps * 0.01)

        if scheduler:
          scheduler = get_linear_schedule_with_warmup(optimizer,
                                                      nr_warm_steps,
                                                      total_steps)
        
        losses = []
        for epoch_idx in range(nr_epochs):
          #Randomize data order 
          data_generator = self.get_data_generator(data, batch_size)

          for batch_idx, (input_batch, label_batch) in tqdm_notebook(enumerate(data_generator), total=nr_batches):

            optimizer.zero_grad()

            # Forward pass
            model_out = self.model.forward(input_ids=input_batch,
                                           labels = label_batch)
             
            loss = model_out.loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            if(batch_idx + 1) % print_freq == 0:
              avg_loss = np.mean(losses[-print_freq])
              print('Epoch: {} | Step: {} | Avg.loss: {:.3f} | lr: {}'.format(
                   epoch_idx+1, batch_idx+1, avg_loss, scheduler.get_last_lr()[0]))

            if (batch_idx + 1) == 18000:
              pickle.dump(self.model, open(save_path, 'wb'))
              pickle.dump(losses, open(losses_path, 'wb'))
              return self.model, losses
      
            if (batch_idx + 1) % checkpoint_freq == 0:
              print('Saving model with test loss of {:.3f}'.format(avg_loss))
              pickle.dump(self.model, open(save_path, 'wb'))
              pickle.dump(losses, open(losses_path, 'wb'))

        return self.model, losses



















        














        