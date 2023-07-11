from deepx.config import args
from deepx.data import TokenDataset
from deepx.model import BertForTokenClassification
from deepx.trainer import TokenTrainer

if __name__ == '__main__':
    schema = ['I', 'O', 'B']
    setattr(args, "schema", schema)
    setattr(args, "num_labels", len(schema))
    trainer = TokenTrainer(
        config=args,
        dataset_base=TokenDataset,
        model_base=BertForTokenClassification
    )
    trainer.init()

    if args.do_train:
        trainer.train_model()

    if args.do_test:
        trainer.load_model(args.output_dir)
        trainer.eval_model(trainer.test_loader)

    if args.do_export:
        trainer.export_model()
