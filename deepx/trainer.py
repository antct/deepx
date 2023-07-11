import datetime
import json
import logging
import math
import os

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, get_scheduler

from deepx.data import DatasetBase, StreamDatasetBase, get_dataloader
from deepx.model.adv_util import AdvModel
from deepx.util import EarlyStopping, ensure_prepared, find_batch_size, on_main_process, seed_everything


class Trainer():

    def __init__(
        self,
        config,
        model_base,
        dataset_base
    ):
        super().__init__()

        self.config = config
        self.model_base = model_base
        self.dataset_base = dataset_base

        self.model_base.trainer = self
        self.dataset_base.trainer = self

        self.accelerator = Accelerator()
        self.components = []

        seed_everything(self.config.seed)

        if self.accelerator.is_main_process and self.config.do_log:
            from accelerate.logging import get_logger
            self.logger = get_logger(__name__)
            logging.basicConfig(
                format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(
                        filename='log/{}.log'.format(
                            datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
                        )
                    ),
                    logging.StreamHandler()
                ]
            )
        if self.accelerator.is_main_process and self.config.do_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()

        self.log('trainer: {}'.format(self.__class__.__name__))
        self.log('dataset: {}'.format(self.dataset_base.__name__))
        self.log('model: {}'.format(self.model_base.__name__))
        self.log('config: {}'.format(json.dumps(vars(self.config), indent=4)))

    def _name(self, attribute):
        for key, value in self.__dict__.items():
            if attribute == value:
                return key
        return None

    @property
    def device(self):
        return self.accelerator.device

    @on_main_process
    def summary(self, tag, scalar_value, global_step):
        if self.config.do_tensorboard:
            self.writer.add_scalar(
                tag=tag,
                scalar_value=scalar_value,
                global_step=global_step
            )

    @on_main_process
    def log(self, msg):
        self.accelerator.print(msg)
        if self.config.do_log:
            self.logger.info(msg=msg)

    def clear(self):
        self.accelerator.clear()
        self.components.clear()

    def prepare(self):
        if not self.components:
            return
        prepare_items = [(self._name(component), component) for component in self.components]
        prepare_keys = [item[0] for item in prepare_items if item[0]]
        prepare_components = [item[1] for item in prepare_items if item[0]]
        prepared_components = self.accelerator.prepare(*prepare_components)
        prepared_components = [prepared_components] if len(prepare_keys) == 1 else prepared_components
        for prepare_key, prepared_component in zip(prepare_keys, prepared_components):
            setattr(self, prepare_key, prepared_component)
        self.components.clear()

    def init(self):
        self.clear()
        self.load_model()
        self.load_dataloader()
        self.load_optimizer_and_scheduler()
        self.prepare()

    def approx_dataloader(self, dataloader):
        if dataloader is None:
            return 0, 0
        if issubclass(self.dataset_base, DatasetBase):
            num_examples = len(dataloader.dataset)
            num_batches = len(dataloader)
            return num_examples, num_batches
        elif issubclass(self.dataset_base, StreamDatasetBase):
            if hasattr(dataloader.dataset, '__len__'):
                num_examples = len(dataloader.dataset)
                num_batches = len(dataloader)
            else:
                batch_size = dataloader.batch_size
                world_size = torch.distributed.get_world_size()
                batch_size = batch_size * world_size
                num_workers = dataloader.num_workers
                num_workers = num_workers if num_workers else 1
                num_examples = len(getattr(dataloader.dataset, 'dataset', dataloader.dataset))
                approx_batch_num = math.ceil(num_examples / batch_size)
                actual_batch_num = math.ceil(approx_batch_num / num_workers)
                num_batches = actual_batch_num * num_workers
            return num_examples, num_batches
        else:
            raise NotImplementedError

    def load_dataloader(self):
        if self.config.do_train:
            self.train_loader = get_dataloader(
                dataset=self.dataset_base,
                example=os.path.join(self.config.data_dir, self.config.train_fn),
                tokenizer=self.tokenizer,
                config=self.config,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                num_workers=self.config.num_workers
            )
            self.log('train data: {}'.format(len(self.train_loader.dataset)))
            self.components.append(self.train_loader)
        if self.config.do_eval:
            self.eval_loader = get_dataloader(
                dataset=self.dataset_base,
                example=os.path.join(self.config.data_dir, self.config.eval_fn),
                tokenizer=self.tokenizer,
                config=self.config,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                num_workers=0
            )
            self.log('eval data: {}'.format(len(self.eval_loader.dataset)))
            self.components.append(self.eval_loader)
        if self.config.do_test:
            self.test_loader = get_dataloader(
                dataset=self.dataset_base,
                example=os.path.join(self.config.data_dir, self.config.test_fn),
                tokenizer=self.tokenizer,
                config=self.config,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                num_workers=0
            )
            self.log('test data: {}'.format(len(self.test_loader.dataset)))
            self.components.append(self.test_loader)

    def load_model(self, model_dir=None, output_attentions=False):
        model_dir = model_dir if model_dir else self.config.model_dir
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            num_labels=self.config.num_labels,
            output_attentions=output_attentions
        )
        self.model = self.model_base.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config=self.model_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config=self.model_config
        )
        if self.config.do_adv:
            if self.config.adv_type not in AdvModel:
                raise NotImplementedError
            self.log('adv model: {}'.format(AdvModel[self.config.adv_type].__name__))
            self.adv_model = AdvModel[self.config.adv_type](self.model)
        self.components.append(self.model)

    def load_optimizer_and_scheduler(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = torch.optim.AdamW(
            params=optimizer_grouped_parameters,
            lr=self.config.lr,
            eps=self.config.eps
        )
        self.global_steps = 0
        self.total_steps = len(self.train_loader) * self.config.num_epochs
        self.total_steps = self.total_steps // self.config.gradient_accumulation_steps
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(self.config.warmup_ratio * self.total_steps),
            num_training_steps=self.total_steps
        )
        self.components.append(self.optimizer)
        self.components.append(self.scheduler)

    @find_batch_size
    @ensure_prepared
    def train_model(self):
        self.stopper = EarlyStopping(
            monitor_key=self.config.stop_key,
            mode=self.config.stop_mode,
            patience=self.config.stop_patience
        )
        self.model.zero_grad()
        for epoch in range(1, self.config.num_epochs + 1):
            if self.config.do_stop and self.stopper.is_stop:
                break
            self.model.train()
            _, num_batches = self.approx_dataloader(self.train_loader)
            epoch_iterator = tqdm(
                iterable=self.train_loader,
                total=num_batches,
                disable=not self.accelerator.is_main_process,
                dynamic_ncols=True
            )
            epoch_iterator.set_description('epoch {}'.format(epoch))
            for _, inputs in enumerate(epoch_iterator):
                self.global_steps += 1

                self.summary('train/lr', self.scheduler.get_last_lr()[0], self.global_steps)
                outputs = self.model(**inputs)

                loss = outputs[0]
                if torch.isnan(loss):
                    raise RuntimeError('found nan in training loss')

                module = getattr(self.model, 'module', self.model)
                loss_dict = getattr(module, 'loss_dict', {})
                epoch_iterator.set_postfix({key: value[0] for key, value in loss_dict.items()})
                for key, value in loss_dict.items():
                    self.summary('train/{}'.format(key), value[0], self.global_steps)

                loss = loss / self.config.gradient_accumulation_steps
                self.accelerator.backward(loss)
                self.summary('train/loss', loss.item(), self.global_steps)

                if self.config.do_adv:
                    if self.config.adv_type in ['fgm', 'fgsm']:
                        self.adv_model.attack()
                        adv_loss = self.model(**inputs)[0]
                        adv_loss = adv_loss / self.config.gradient_accumulation_steps
                        self.accelerator.backward(adv_loss)
                        self.adv_model.restore()
                    elif self.config.adv_type in ['pgd']:
                        adv_k = 3
                        self.adv_model.backup_grad()
                        for k in range(adv_k):
                            self.adv_model.attack(backup=(k == 0))
                            if k != adv_k - 1:
                                self.model.zero_grad()
                            else:
                                self.adv_model.restore_grad()
                            adv_loss = self.model(**inputs)[0]
                            adv_loss = adv_loss / self.config.gradient_accumulation_steps
                            self.accelerator.backward(adv_loss)
                        self.adv_model.restore()
                    else:
                        raise NotImplementedError

                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                if self.global_steps % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                if self.config.do_eval and self.global_steps % self.config.eval_steps == 0:
                    metrics = self.eval_model(self.eval_loader)
                    self.stopper.step(metrics, self.global_steps)
                    if self.config.do_stop:
                        if self.stopper.is_best:
                            self.log('best model')
                            self.save_model()
                        if self.stopper.is_stop:
                            self.log('early stop')
                            break
                    self.model.train()

            if self.config.save_epochs > 0 and epoch % self.config.save_epochs == 0:
                self.save_model(subdir='epoch-{}'.format(epoch))

        if self.config.do_eval and self.config.do_stop:
            metrics = self.eval_model(self.eval_loader)
            self.stopper.step(metrics, self.global_steps)
            if self.stopper.is_best:
                self.log('best model')
                self.save_model()
        else:
            self.save_model()

        self.accelerator.wait_for_everyone()
        return self.stopper.best_metrics

    @ensure_prepared
    def eval_model(self, eval_loader, return_metrics=True):
        from scipy.special import softmax
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        self.model.eval()
        num_examples, num_batches = self.approx_dataloader(eval_loader)
        epoch_iterator = tqdm(
            iterable=eval_loader,
            disable=not self.accelerator.is_main_process,
            total=num_batches
        )
        epoch_iterator.set_description('eval')
        sample_seen = 0
        eval_losses, preds, trues, scores = [], [], [], []
        with torch.no_grad():
            for step, inputs in enumerate(epoch_iterator):
                labels = inputs['labels']
                outputs = self.model(**inputs)
                step_eval_loss, logits = outputs[:2]
                eval_losses.append(step_eval_loss.item())

                logits, labels = self.accelerator.gather((logits, labels))

                if self.accelerator.num_processes > 1:
                    if step == num_batches - 1:
                        logits = logits[:num_examples - sample_seen]
                        labels = labels[:num_examples - sample_seen]
                    else:
                        sample_seen += logits.shape[0]

                logits = logits.cpu().numpy()
                labels = labels.cpu().numpy()

                preds.extend(np.argmax(logits, axis=-1).tolist())
                trues.extend(labels.tolist())
                scores.extend(softmax(logits, axis=-1).tolist())

        if not return_metrics:
            return {}

        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(os.path.join(self.config.output_dir, self.config.output_fn), 'w+') as wf:
                for true, pred, score in zip(trues, preds, scores):
                    wf.write(json.dumps({
                        'label': true,
                        'prediction': pred,
                        'score': score
                    }) + '\n')

        metrics = {}

        acc = accuracy_score(trues, preds)
        auc = roc_auc_score(trues, scores, multi_class='ovr')
        macro_p = precision_score(trues, preds, average='macro', zero_division=0)
        macro_r = recall_score(trues, preds, average='macro', zero_division=0)
        macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        micro_p = precision_score(trues, preds, average='micro', zero_division=0)
        micro_r = recall_score(trues, preds, average='micro', zero_division=0)
        micro_f1 = f1_score(trues, preds, average='micro', zero_division=0)

        self.summary('eval/loss', sum(eval_losses) / len(eval_losses), self.global_steps)
        self.summary('eval/acc', acc, self.global_steps)
        self.summary('eval/auc', auc, self.global_steps)

        metrics['step'] = self.global_steps
        metrics['acc'] = acc
        metrics['auc'] = auc
        metrics['macro'] = {
            'p': macro_p,
            'r': macro_r,
            'f1': macro_f1
        }
        metrics['micro'] = {
            'p': micro_p,
            'r': micro_r,
            'f1': micro_f1
        }

        self.log(json.dumps(metrics, indent=4))
        return metrics

    def save_model(self, subdir=''):
        output_dir = os.path.join(self.config.output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            save_directory=output_dir,
            save_function=self.accelerator.save
        )
        self.tokenizer.save_pretrained(
            save_directory=output_dir,
            save_function=self.accelerator.save
        )
        self.log('save model to {}'.format(output_dir))

    @on_main_process
    def export_model(self):
        from collections import OrderedDict
        from pathlib import Path

        from transformers.onnx import OnnxConfig, export
        self.accelerator.wait_for_everyone()

        class BertOnnxConfig(OnnxConfig):
            @property
            def inputs(self):
                return OrderedDict(
                    [
                        ('input_ids', {0: 'batch', 1: 'sequence'}),
                        ('attention_mask', {0: 'batch', 1: 'sequence'}),
                    ]
                )

            @property
            def outputs(self):
                return OrderedDict({'logits': {0: 'batch', 1: 'sequence'}})

        onnx_config = BertOnnxConfig(self.model_config)
        os.makedirs(os.path.join(self.config.output_dir, 'export'), exist_ok=True)
        onnx_path = Path(os.path.join(self.config.output_dir, 'export/model.onnx'))
        unwrapped_model = self.accelerator.unwrap_model(self.model).cpu()
        unwrapped_model.is_export = True
        onnx_inputs, onnx_outputs = export(
            preprocessor=self.tokenizer,
            model=unwrapped_model,
            config=onnx_config,
            opset=onnx_config.default_onnx_opset,
            output=onnx_path
        )
        self.log('onnx inputs: {}'.format(onnx_inputs))
        self.log('onnx outputs: {}'.format(onnx_outputs))

    @ensure_prepared
    def infer_model(self, data, return_dict=False):
        from scipy.special import expit, softmax
        self.model.eval()
        pred_loader = get_dataloader(
            dataset=self.dataset_base,
            example=data,
            tokenizer=self.tokenizer,
            config=self.config,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0
        )
        pred_loader = self.accelerator.prepare(pred_loader)
        epoch_iterator = tqdm(
            iterable=pred_loader,
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True
        )
        epoch_iterator.set_description('test')
        sample_seen = 0
        preds, scores, representations = [], [], []
        tokens, attentions = [], []
        with torch.no_grad():
            for step, inputs in enumerate(epoch_iterator):
                outputs = self.model(**inputs)
                logits = outputs[0]
                embeddings = outputs[1]
                if self.config.server_bertviz:
                    tokens.extend([self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])])
                    attentions.extend([outputs[-1]])
                logits = self.accelerator.gather(logits)
                if self.accelerator.num_processes > 1:
                    if step == len(pred_loader) - 1:
                        logits = logits[:len(pred_loader.dataset) - sample_seen]
                    else:
                        sample_seen += logits.shape[0]
                logits = logits.cpu().numpy()
                if logits.shape[1] == 1:
                    scores.extend(expit(logits).tolist())
                else:
                    preds.extend(np.argmax(logits, axis=-1).tolist())
                    scores.extend(softmax(logits, axis=-1).tolist())
                representations.extend(embeddings.tolist())

        if not return_dict:
            if self.config.server_bertviz:
                return preds, scores, representations, tokens, attentions
            else:
                return preds, scores, representations
        else:
            if self.config.server_bertviz:
                return {
                    'prediction': preds,
                    'score': scores,
                    'embedding': representations,
                    'token': tokens,
                    'attention': attentions
                }
            else:
                return {
                    'prediction': preds,
                    'score': scores,
                    'embedding': representations
                }

    def cross_train_model(self, k_fold=5):
        from cleanlab.filter import find_label_issues
        from scipy.special import softmax

        self.log("prepare {}-fold dataset".format(k_fold))
        if self.accelerator.is_main_process:
            train_fn = os.path.join(self.config.data_dir, self.config.train_fn)
            data_len = 0
            with open(train_fn, 'r') as f:
                for _ in f:
                    data_len += 1
            slice_len = int(data_len / k_fold) + 1
            slice_idxs = [[i, min(i + slice_len, data_len)] for i in range(0, data_len, slice_len)]

            os.makedirs(os.path.join(self.config.output_dir, 'cross'), exist_ok=True)
            for fold in range(1, k_fold + 1):
                cross_train_fn = os.path.join(self.config.output_dir, 'cross/train_fold_{}'.format(fold))
                cross_eval_fn = os.path.join(self.config.output_dir, 'cross/eval_fold_{}'.format(fold))
                cross_test_fn = os.path.join(self.config.output_dir, 'cross/test_fold_{}'.format(fold))

                data_start, data_end = slice_idxs[fold - 1]
                with open(cross_train_fn, 'w+') as train_wf, \
                        open(cross_eval_fn, 'w+') as eval_wf, open(cross_test_fn, 'w+') as test_wf:
                    with open(train_fn, 'r') as f:
                        for idx, line in enumerate(f):
                            if data_start <= idx < data_end:
                                eval_wf.write(line)
                                test_wf.write(line)
                            else:
                                train_wf.write(line)

        self.accelerator.wait_for_everyone()

        for fold in range(1, k_fold + 1):
            self.config.data_dir = self.config.output_dir
            self.config.train_fn = 'cross/train_fold_{}'.format(fold)
            self.config.eval_fn = 'cross/eval_fold_{}'.format(fold)
            self.config.test_fn = 'cross/test_fold_{}'.format(fold)
            self.config.output_fn = 'cross/pred_fold_{}'.format(fold)
            self.config.do_train = True
            self.config.do_eval = True
            self.config.do_test = True
            self.config.do_stop = False
            # FIXME: dataloader stuck
            self.config.num_workers = 0
            self.init()
            self.train_model()
            self.eval_model(self.test_loader)

        if not self.accelerator.is_main_process:
            return

        self.log("conduct {}-fold cross validation".format(k_fold))

        idxs, labels, pred_probs = [], [], []
        for fold in range(1, k_fold + 1):
            cross_pred_fn = os.path.join(self.config.output_dir, 'cross/pred_fold_{}'.format(fold))
            data_start, data_end = slice_idxs[fold - 1]
            idxs.extend([idx for idx in range(data_start, data_end)])
            with open(cross_pred_fn, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    label, score = line['label'], softmax(line['score'])
                    labels.append(label)
                    pred_probs.append(score)

        assert len(idxs) == len(labels) == len(pred_probs)

        labels = np.array(labels)
        pred_probs = np.array(pred_probs)
        label_issues = find_label_issues(
            labels=labels,
            pred_probs=pred_probs
        )
        check_result_fn = os.path.join(self.config.output_dir, 'cross/check.json')
        with open(check_result_fn, 'w+') as wf:
            for idx, issue in zip(idxs, label_issues):
                wf.write(json.dumps({
                    'idx': idx,
                    'issue': bool(issue)
                }, ensure_ascii=False) + '\n')

        issue_result_fn = os.path.join(self.config.output_dir, 'cross/issue.json')
        clean_result_fn = os.path.join(self.config.output_dir, 'cross/clean.json')
        issue_cnt = 0
        clean_cnt = 0
        with open(train_fn, 'r') as f1, open(check_result_fn, 'r') as f2:
            with open(issue_result_fn, 'w+') as wf1, open(clean_result_fn, 'w+') as wf2:
                for idx, (line, check) in enumerate(zip(f1, f2)):
                    check = json.loads(check)
                    assert idx == check['idx']
                    if check['issue']:
                        issue_cnt += 1
                        wf1.write(line)
                    else:
                        clean_cnt += 1
                        wf2.write(line)

        issue_ratio = issue_cnt / (issue_cnt + clean_cnt)
        return issue_ratio

    def cross_train_loop(self, k_fold=5, eps=0.001, max_loops=10):
        import shutil
        if self.accelerator.is_main_process:
            os.makedirs(os.path.join(self.config.output_dir, 'cross'), exist_ok=True)
            train_fn = os.path.join(self.config.data_dir, self.config.train_fn)
            raw_fn = os.path.join(self.config.output_dir, 'cross/raw.json')
            with open(raw_fn, 'w+') as wf:
                with open(train_fn, 'r') as f:
                    for _, line in enumerate(f):
                        wf.write(line)

        self.accelerator.wait_for_everyone()

        loop = 0
        issue_ratio = 1.0
        output_dir = self.config.output_dir
        while issue_ratio > eps and loop < max_loops:
            self.config.data_dir = '{}/cross'.format(output_dir)
            self.config.output_dir = output_dir
            self.config.train_fn = 'raw.json'
            ratio = self.cross_train_model(k_fold=k_fold)
            # HACK: ratio returns only main process
            ratio = ratio if ratio else 1.0
            ratio = torch.tensor(ratio).float().to(self.accelerator.device)
            if self.accelerator.is_main_process:
                clean_result_fn = os.path.join(self.config.output_dir, 'cross/clean.json')
                shutil.copyfile(clean_result_fn, raw_fn)
            loop += 1
            ratio = self.accelerator.gather(ratio)
            issue_ratio = min(ratio.cpu().numpy().tolist())
            self.log('{}-loop issue ratio: {}'.format(loop, issue_ratio))


class TokenTrainer(Trainer):

    def __init__(self, config, model_base, dataset_base):
        super().__init__(
            config=config,
            model_base=model_base,
            dataset_base=dataset_base
        )

    @ensure_prepared
    def eval_model(self, eval_loader, return_metrics=True):
        from seqeval.metrics import accuracy_score, f1_score, recall_score
        self.model.eval()
        num_examples, num_batches = self.approx_dataloader(eval_loader)
        epoch_iterator = tqdm(
            iterable=eval_loader,
            disable=not self.accelerator.is_main_process,
            total=num_batches
        )
        epoch_iterator.set_description('eval')
        sample_seen = 0
        preds, trues = [], []
        with torch.no_grad():
            for step, inputs in enumerate(epoch_iterator):
                labels = inputs['labels']
                logits = self.model.module.decode(**inputs)
                logits, labels = self.accelerator.gather((logits, labels))

                if self.accelerator.num_processes > 1:
                    if step == num_batches - 1:
                        logits = logits[:num_examples - sample_seen]
                    else:
                        sample_seen += logits.shape[0]

                logits = logits.cpu().numpy()
                labels = labels.cpu().numpy()

                preds.extend(logits.tolist())
                trues.extend(labels.tolist())

        if not return_metrics:
            return {}

        id2label = {i: self.config.schema[i] for i in range(len(self.config.schema))}

        predictions = [
            [id2label[token_pred] for (token_pred, token_true) in zip(seq_pred, seq_true) if token_true != -100]
            for seq_pred, seq_true in zip(preds, trues)
        ]
        references = [
            [id2label[token_true] for (_, token_true) in zip(seq_pred, seq_true) if token_true != -100]
            for seq_pred, seq_true in zip(preds, trues)
        ]

        metrics = {}
        metrics['step'] = self.global_steps
        metrics['acc'] = accuracy_score(references, predictions)
        metrics['recall'] = recall_score(references, predictions)
        metrics['f1'] = f1_score(references, predictions)

        self.summary('eval/acc', metrics['acc'], self.global_steps)
        self.summary('eval/f1', metrics['f1'], self.global_steps)

        self.log(json.dumps(metrics, indent=4))
        return metrics

    @ensure_prepared
    def infer_model(self, data, return_dict=False):
        self.model.eval()
        pred_loader = get_dataloader(
            dataset=self.dataset_base,
            example=data,
            tokenizer=self.tokenizer,
            config=self.config,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0
        )
        pred_loader = self.accelerator.prepare(pred_loader)
        epoch_iterator = tqdm(
            iterable=pred_loader,
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True
        )
        epoch_iterator.set_description('test')

        sample_seen = 0
        preds = []
        with torch.no_grad():
            for step, inputs in enumerate(epoch_iterator):
                labels = inputs['labels']
                logits = self.model.module.decode(**inputs)
                logits, labels = self.accelerator.gather((logits, labels))

                if self.accelerator.num_processes > 1:
                    if step == len(pred_loader) - 1:
                        logits = logits[:len(pred_loader.dataset) - sample_seen]
                    else:
                        sample_seen += logits.shape[0]

                logits = logits.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                id2label = {i: self.config.schema[i] for i in range(len(self.config.schema))}
                predictions = [
                    [id2label[token_pred] for (token_pred, token_true) in zip(seq_pred, seq_true) if token_true != -100]
                    for seq_pred, seq_true in zip(logits, labels)
                ]
                preds.extend(predictions)

        if not return_dict:
            return (preds, )
        else:
            return {
                'prediction': preds
            }
