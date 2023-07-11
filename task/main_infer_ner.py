from deepx.config import args
from deepx.data import TokenInferDataset
from deepx.model import BertForTokenClassification
from deepx.trainer import TokenTrainer

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
    data = [
        [["杭", "州", "是", "一", "个", "美", "丽", "的", "城", "市"]]
    ]
    res = trainer.infer_model(data=data, return_dict=True)
    trainer.log(res)
