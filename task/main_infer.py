from deepx.config import args
from deepx.data import InferDataset
from deepx.model import BertForSequenceClassification
from deepx.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(
        config=args,
        dataset_base=InferDataset,
        model_base=BertForSequenceClassification
    )
    trainer.load_model()
    data = [
        ['杭州', '浙江']
    ]
    res = trainer.infer_model(data=data, return_dict=True)
    trainer.log(res)
