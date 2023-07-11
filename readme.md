## deepx

### 目录结构
```
├── deepx
│   ├── config.py                       # 运行配置
│   ├── dataloader                      # 数据加载
│   ├── model                           # 模型文件
│   ├── trainer.py                      # 训练文件
│   └── util.py                         # 辅助工具
├── readme.md                           # 使用文档
├── requirements.txt                    # 安装配置
├── script
│   └── run.sh                          # 运行脚本
├── task
│   └── main.py                         # 任务入口
└── yaml
    └── ddp.yaml                        # DDP配置
```

### 快速开始

```bash
pip install -r requirements.txt
```

```python
class Dataset(DatasetBase):

    def __init__(self, example, tokenizer, config):
        super().__init__(
            example=example,
            tokenizer=tokenizer,
            config=config
        )

    def line_fn(self, line):
        # 文件每行数据处理逻辑

    def data_fn(self, data):
        # 批次数据自定义聚合逻辑
```

```python
class BertForSequenceClassification(BertModelBase):

    def __init__(self, config, freeze_bert=False):
        super().__init__(
            config=config,
            freeze_bert=freeze_bert
        )
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None
    ):
        # 模型前向传播逻辑
```

```python
from deepx.config import args
from deepx.data import Dataset, StreamDataset
from deepx.model import BertForSequenceClassification
from deepx.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(
        # 训练配置
        config=args,
        # 数据基类
        dataset_base=Dataset/StreamDataset,
        # 模型基类
        model_base=BertForSequenceClassification
    )
    # 初始化数据/模型/优化器
    trainer.init()
    # 模型训练/评估
    if args.do_train:
        trainer.train_model()
    # 模型预测
    if args.do_test:
        trainer.load_model(args.output_dir)
        trainer.eval_model(trainer.test_loader)
    # 模型导出
    if args.do_export:
        trainer.export_model()
```

### 运行脚本

```bash
source ${BASH_SOURCE%/*}/env.sh
launch task/${任务入口} \
    --do_train \
    --do_eval \
    --do_test \
    --do_stop \
    --do_export \
    --input_qd \
    --input_format json \
    --data_dir ${数据路径} \
    --model_dir ${模型路径} \
    --output_dir checkpoint/model \
    --train_fn train.json \
    --eval_fn eval.json \
    --test_fn test.json \
    --max_seq_length 64 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --num_epochs 2
```