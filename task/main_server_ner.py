import json

from flask import Flask, request

from deepx.config import args
from deepx.data import TokenInferDataset
from deepx.model import BertForTokenClassification
from deepx.trainer import TokenTrainer

app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    data = request.get_data().decode('utf-8')
    data = json.loads(data)['querys']
    infer_res = trainer.infer_model(data=data, return_dict=True)
    model_res = [infer_res['prediction']]
    return json.dumps({'code': 0, 'result': model_res, 'msg': 'ok'}, ensure_ascii=False)


if __name__ == '__main__':
    schema = ['I', 'O', 'B']
    setattr(args, "schema", schema)
    setattr(args, "num_labels", len(schema))
    trainer = TokenTrainer(
        config=args,
        dataset_base=TokenInferDataset,
        model_base=BertForTokenClassification
    )
    trainer.load_model()

    app.run(host='0.0.0.0', port=args.server_port, debug=False)
