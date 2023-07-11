import json
import os

import faiss
import numpy as np
from flask import Flask, request

from deepx.config import args
from deepx.data import InferDataset
from deepx.model import BertForSequenceClassification
from deepx.trainer import Trainer

app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    data = request.get_data().decode('utf-8')
    data = json.loads(data)['query']
    infer_res = trainer.infer_model(data=data, return_dict=True)
    model_res = [infer_res['score']]
    return json.dumps({'code': 0, 'result': model_res, 'msg': 'ok'}, ensure_ascii=False)


@app.route('/faiss', methods=['GET'])
def search():
    top_k = int(request.args.get('top_k', '5'))
    query = request.args.get('query')
    title = request.args.get('title')
    infer_res = trainer.infer_model(data=[[query, title]], return_dict=True)
    embedding = infer_res['embedding']
    D, P = faiss_index.search(np.array(embedding).astype('float32'), top_k)
    D = D.tolist()
    P = P.tolist()
    search_res = []
    for d, p in zip(D, P):
        search_res.append([
            {'query': faiss_kv[str(pp)][0],
                'title': faiss_kv[str(pp)][1],
                'label': faiss_kv[str(pp)][2],
                'distance': dd}
            for (pp, dd) in zip(p, d)
        ])
    return json.dumps({'code': 0, 'result': search_res, 'msg': 'ok'}, ensure_ascii=False)


@app.route('/bertviz', methods=['GET'])
def viz():
    from bertviz import head_view
    layer = int(request.args.get('layer', '5'))
    query = request.args.get('query')
    title = request.args.get('title')
    infer_res = trainer.infer_model(data=[[query, title]], return_dict=True)
    token, attention = infer_res['token'][0], infer_res['attention'][0]
    padding_pos = token.index('[PAD]')
    token = token[:padding_pos]
    attention = [layer_attention[:, :, :padding_pos, :padding_pos] for layer_attention in attention]
    html_head_view = head_view(
        attention=attention,
        tokens=token,
        sentence_b_start=token.index('[SEP]') + 1,
        layer=layer,
        html_action='return'
    )
    return html_head_view.data


if __name__ == '__main__':
    trainer = Trainer(
        config=args,
        dataset_base=InferDataset,
        model_base=BertForSequenceClassification
    )
    trainer.load_model(output_attentions=args.server_bertviz)

    if args.server_faiss:
        index_type = 'flat'
        trainer.log('loading faiss index ...')
        faiss_index = faiss.read_index(os.path.join(trainer.config.output_dir, 'faiss/{}.index'.format(index_type)))
        trainer.log('loading faiss kv ...')
        faiss_kv = json.load(open(os.path.join(trainer.config.output_dir, 'faiss/kv.json'), 'r'))

    app.run(host='0.0.0.0', port=args.server_port, debug=False)
