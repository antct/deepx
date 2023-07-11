from deepx.config import args
from deepx.data import Dataset
from deepx.model import BertForSequenceClassification
from deepx.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(
        config=args,
        dataset_base=Dataset,
        model_base=BertForSequenceClassification
    )
    trainer.init()

    if args.do_train:
        trainer.train_model()

    if args.do_test:
        trainer.load_model(args.output_dir)
        trainer.eval_model(trainer.test_loader)

    if args.do_export:
        trainer.export_model()
