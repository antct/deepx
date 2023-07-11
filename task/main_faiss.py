import json
import os

import faiss
import numpy as np
from tqdm import tqdm

from deepx.config import args
from deepx.data import InferDataset
from deepx.model import BertForSequenceClassification
from deepx.trainer import Trainer


def build_index(trainer, index_type='flat'):
    os.makedirs(os.path.join(trainer.config.output_dir, 'faiss'), exist_ok=True)
    faiss_fn = os.path.join(trainer.config.output_dir, 'faiss/{}.index'.format(index_type))
    kv_fn = os.path.join(trainer.config.output_dir, 'faiss/kv.json')
    train_fn = os.path.join(trainer.config.data_dir, trainer.config.train_fn)
    buffer, buffer_size = [], 1024 * 100
    feature, kv = [], {}
    # TODO 考虑引入增量PCA来降维
    if index_type == 'flat':
        faiss_index = faiss.index_factory(768, 'Flat', faiss.METRIC_L2)
    elif index_type == 'hnsw':
        faiss_index = faiss.index_factory(768, 'HNSW8', faiss.METRIC_L2)
    else:
        raise NotImplementedError

    def batch_store():
        nonlocal buffer
        nonlocal feature
        assert len(buffer) == len(feature)
        res = trainer.infer_model(data=buffer, return_dict=True)
        embeddings = res['embedding']
        for feature, embedding in zip(feature, embeddings):
            idx, query, title, label = feature
            feature = np.array([embedding]).astype('float32')
            # hnsw添加索引慢, flat索引2000w预计1.5小时
            faiss_index.add(feature)
            kv[str(idx)] = [query, title, label]
        buffer = []
        feature = []
    cnt = 0
    with open(train_fn, 'r') as f:
        for idx, _ in enumerate(f):
            cnt += 1
    qds = set()
    idx = 0
    with open(train_fn, 'r') as f:
        for _, line in tqdm(enumerate(f), total=cnt):
            line = json.loads(line.rstrip())
            # 训练数据处理逻辑
            # TODO 实现text输入逻辑
            if (line['query'], line['title']) not in qds:
                buffer.append((line['query'], line['title']))
                feature.append((idx, line['query'], line['title'], line['label']))
                qds.add((line['query'], line['title']))
                idx += 1
            # 处理逻辑结束
            if len(buffer) >= buffer_size:
                batch_store()
        if len(feature) > 0:
            batch_store()
    faiss.write_index(faiss_index, faiss_fn)
    with open(kv_fn, 'w+') as wf:
        wf.write(json.dumps(kv))


if __name__ == '__main__':
    trainer = Trainer(
        config=args,
        dataset_base=InferDataset,
        model_base=BertForSequenceClassification
    )
    trainer.load_model()
    build_index(trainer)
