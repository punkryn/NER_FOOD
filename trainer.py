import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import compute_metrics, get_labels, show_report, MODEL_CLASSES

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

#from TorchCRF import CRF

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.FOOD_score = 0
        self.FOOD_list = []

        self.label_lst = get_labels(args)
        print('label', self.label_lst)
        self.num_labels = len(self.label_lst)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=args.task,
                                                        id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})
        self.model = self.model_class.from_pretrained(args.model_name_or_path, config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        #print((self.train_dataset))
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        best_f1_score = 0
        best_FOOD_score = 0
        out_label_list = []
        preds_list = []
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results, out_label_list, preds_list, diff_true, diff_pred, equal_true, equal_pred = self.evaluate("test")  # Only test set available for NSMC
                        best_f1_score = results['f1'] if best_f1_score > results['f1'] else results['f1']
                        print('results', results)
                        print('best f1 score', best_f1_score)                        
                        #label_lst = self.label_lst
                        label_index_to_print = list(range(2, 24))
                        print('ret', type(out_label_list), out_label_list)
                        print('ret', type(preds_list), preds_list)
                        print(2)
                        #self.plot_confusion_matrix(out_label_list, preds_list, self.label_lst, label_index_to_print, False, 'Confusion matrix, without normalization')
                        self.plot_confusion_matrix(y_true=out_label_list, y_pred=preds_list, classes=self.label_lst, labels=label_index_to_print, normalize=False, title='Confusion matrix, without normalization')
                        plt.savefig('confusion_matrix.png')
                        
                        if self.FOOD_score >= best_FOOD_score:
                          best_FOOD_score = self.FOOD_score
                          with open('./true.txt', 'w', encoding='utf-8') as ttw:
                            ttw.write('best_FOOD_score ')
                            ttw.write(str(best_FOOD_score))
                            ttw.write('\n')
                            for i, item in enumerate(equal_true):
                              for tag in item:
                                ttw.write(tag + ' ')
                              ttw.write('\n')

                              for tag in equal_pred[i]:
                                ttw.write(tag + ' ')
                              ttw.write('\n\n')

                          with open('./miss.txt', 'w', encoding='utf-8') as mtw:
                            mtw.write('best_FOOD_score ')
                            mtw.write(str(best_FOOD_score))
                            mtw.write('\n')
                            for i, item in enumerate(diff_true):
                              for tag in item:
                                mtw.write(tag + ' ')
                              mtw.write('\n')

                              for tag in diff_pred[i]:
                                mtw.write(tag + ' ')
                              mtw.write('\n\n')

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()
                        print('best f1 score', best_f1_score)                        
                        #label_lst = self.label_lst
                        label_index_to_print = list(range(2, 24))
                        self.plot_confusion_matrix(y_true=out_label_list, y_pred=preds_list, classes=self.label_lst, labels=label_index_to_print, normalize=False, title='Confusion matrix, without normalization')
                        #self.plot_confusion_matrix(out_label_list, preds_list, self.label_lst, label_index_to_print, False, 'Confusion matrix, without normalization')
                        plt.savefig('confusion_matrix.png')

                        if self.FOOD_score >= best_FOOD_score:
                          best_FOOD_score = self.FOOD_score
                          with open('./true.txt', 'w', encoding='utf-8') as ttw:
                            ttw.write('best_FOOD_score ')
                            ttw.write(str(best_FOOD_score))
                            ttw.write('\n')
                            for i, item in enumerate(equal_true):
                              for tag in item:
                                ttw.write(tag + ' ')
                              ttw.write('\n')

                              for tag in equal_pred[i]:
                                ttw.write(tag + ' ')
                              ttw.write('\n\n')

                          with open('./miss.txt', 'w', encoding='utf-8') as mtw:
                            mtw.write('best_FOOD_score ')
                            mtw.write(str(best_FOOD_score))
                            mtw.write('\n')
                            for i, item in enumerate(diff_true):
                              for tag in item:
                                mtw.write(tag + ' ')
                              mtw.write('\n')

                              for tag in diff_pred[i]:
                                mtw.write(tag + ' ')
                              mtw.write('\n\n')

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        trues = None
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                trues = inputs['input_ids'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                trues = np.append(trues, inputs['input_ids'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        print('preds', preds)
        print('out_label_ids', out_label_ids)
        print('trues', trues)

        # Slot result
        preds = np.argmax(preds, axis=2)
        trues = np.argmax(trues, axis=1)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])
        print('list1', out_label_list)
        print('list2', preds_list)
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result
        sr = show_report(out_label_list, preds_list)
        print('sr', type(sr), sr)
        sr_list = sr.split('\n')
        print('srl', sr_list)
        for sen in sr_list:
          tag_list = sen.split()
          print('tl', tag_list)
          if len(tag_list) != 0:
            if tag_list[0] == 'FOOD':
              self.FOOD_list = tag_list
              self.FOOD_score = float(tag_list[3])
        print('food-list', self.FOOD_list)
        print('food-score', self.FOOD_score)
        print('return', type(results))
        print('return', type(out_label_list), out_label_list)
        print('return', type(preds_list), preds_list)
        print('return', type(trues), trues)
        print(3)
        out_label_list = np.array(out_label_list).flatten().tolist()
        preds_list = np.array(preds_list).flatten().tolist()

        diff_true = []
        diff_pred = []

        equal_true = []
        equal_pred = []
        for i, seq in enumerate(out_label_list):
          if 'FOOD-B' in preds_list[i]:
            if preds_list[i] != seq:
              diff_true.append(seq)
              diff_pred.append(preds_list[i])
            else:
              equal_true.append(seq)
              equal_pred.append(preds_list[i])

        tmp = []
        for seq in out_label_list:
          tmp += seq
        out_label_list = tmp
        
        tmp = []
        for seq in preds_list:
          tmp += seq
        preds_list = tmp

        tmp = []
        for tag in out_label_list:
          tmp.append(self.label_lst.index(tag))
        out_label_list = tmp

        tmp = []
        for tag in preds_list:
          tmp.append(self.label_lst.index(tag))
        preds_list = tmp

        print('oll', out_label_list)
        print('pl', preds_list)
        print('diff_ture', diff_true)
        print('diff_pred', diff_pred)
        return results, out_label_list, preds_list, diff_true, diff_pred, equal_true, equal_pred

    
    def plot_confusion_matrix(self, y_true, y_pred, classes, labels, normalize=False, title=None, cmap=plt.cm.Blues):
      print(1)
    # """
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    # """
      if not title:
          if normalize:
              title = 'Normalized confusion matrix'
          else:
              title = 'Confusion matrix, without normalization'

      # Compute confusion matrix
      print(y_true, y_pred, labels)
      print(type(y_true), type(y_pred), type(labels))
      cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
      
      # Only use the labels that appear in the data

      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
      else:
          print('Confusion matrix, without normalization')

      # --- plot 크기 조절 --- #
      plt.rcParams['savefig.dpi'] = 200
      plt.rcParams['figure.dpi'] = 200
      plt.rcParams['figure.figsize'] = [20, 20]  # plot 크기
      plt.rcParams.update({'font.size': 10})
      # --- plot 크기 조절 --- #

      fig, ax = plt.subplots()
      im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

      # --- bar 크기 조절 --- #
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(im, cax=cax)
      # --- bar 크기 조절 --- #
      # ax.figure.colorbar(im, ax=ax)

      # We want to show all ticks...
      ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes[2:], yticklabels=classes[2:],
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

      # Rotate the tick labels and set their alignment.
      plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
              rotation_mode="anchor")

      # Loop over data dimensions and create text annotations.
      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i in range(cm.shape[0]):
          for j in range(cm.shape[1]):
              ax.text(j, i, format(cm[i, j], fmt),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")
      fig.tight_layout()
      return ax

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
