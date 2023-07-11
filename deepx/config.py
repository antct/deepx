import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--do_debug', action='store_true', help='调试模式(仅使用500条数据)')
parser.add_argument('--do_train', action='store_true', help='模型训练(train_fn)')
parser.add_argument('--do_eval', action='store_true', help='训练阶段评估模型(eval_fn)')
parser.add_argument('--do_test', action='store_true', help='模型测试(test_fn)')
parser.add_argument('--do_export', action='store_true', help='模型导出(onnx)')
parser.add_argument('--do_stop', action='store_true', help='训练阶段早停')
parser.add_argument('--do_log', action='store_true', help='输出日志')
parser.add_argument('--do_tensorboard', action='store_true', help='tensorboard')
parser.add_argument('--do_adv', action='store_true', help='对抗训练')

parser.add_argument('--model_dir', type=str, default='bert-wwm-ext', help="初始加载模型目录/预训练模型目录")
parser.add_argument('--output_dir', type=str, default='checkpoint/model', help="输出模型目录")
parser.add_argument('--data_dir', type=str, default='data', help="数据目录")
parser.add_argument('--train_fn', type=str, default='train.txt', help="训练集")
parser.add_argument('--eval_fn', type=str, default='test.txt', help="验证集")
parser.add_argument('--test_fn', type=str, default='test.txt', help="测试集")
parser.add_argument('--output_fn', type=str, default='pred.txt', help="预测集合(无标签)")

parser.add_argument('--input_qd', action='store_true', help="样本格式(query/query-doc)")
parser.add_argument('--input_format', type=str, default='text', help="输入格式(text/json)")
parser.add_argument('--max_seq_length', type=int, default=64, help="最大序列长度")

parser.add_argument('--seed', type=int, default=42, help="随机种子")
parser.add_argument('--num_labels', type=int, default=5, help="类别数目")
parser.add_argument('--num_epochs', type=int, default=2, help="训练轮次")
parser.add_argument('--lr', type=float, default=2e-5, help="学习率")
parser.add_argument('--eps', type=float, default=1e-8, help="最小变化学习率")
parser.add_argument('--warmup_ratio', type=float, default=0.1, help="学习率预热比例")
parser.add_argument('--weight_decay', type=float, default=1e-2, help="权重衰减系数")
parser.add_argument('--max_grad_norm', type=float, default=1.0, help="最大梯度范数")
parser.add_argument('--lr_scheduler_type', type=str, default='linear', help="学习率衰减模式")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="梯度累积步数")
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help="单卡训练批次大小")
parser.add_argument('--per_device_eval_batch_size', type=int, default=128, help="单卡评估批次大小")
parser.add_argument('--eval_steps', type=int, default=1000, help="模型评估步数周期")
parser.add_argument('--save_epochs', type=int, default=-1, help="模型保存轮次周期")
parser.add_argument('--num_workers', type=int, default=16, help="并发数据读取")

parser.add_argument('--adv_type', type=str, default='fgm', help='对抗训练方法')

parser.add_argument('--stop_key', type=str, default='acc', help="早停监控指标")
parser.add_argument('--stop_mode', type=str, default='max', help="早停指标比较模式")
parser.add_argument('--stop_patience', type=int, default=100, help="早停最大耐心值")

parser.add_argument('--server_port', type=int, default=6000, help="flask服务端口")
parser.add_argument('--server_faiss', action='store_true', help="flask服务(faiss)")
parser.add_argument('--server_bertviz', action='store_true', help="flask服务(bertviz)")

args, unknown = parser.parse_known_args()
