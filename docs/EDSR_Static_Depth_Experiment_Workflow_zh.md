# EDSR静态减深训练、验证与测试工作流

## 一次性环境设置

每台机器只设置一次：

```bash
cp .env.example .env
```

RTX 4060：

```dotenv
SR_DATA_ROOT=/home/jww/data
```

Featurize：

```dotenv
SR_DATA_ROOT=/home/featurize/data
```

`.env`不进入Git。程序自动加载它，并把实际展开的路径写入实验目录中的resolved YAML。

## 公共入口

```bash
python codes/train.py -opt <train.yml>
python codes/test.py  -opt <test.yml>
python codes/run.py   -opt <run-plan.yml>
```

模型由YAML中的`model`和`network_G.which_model_G`选择，公开脚本不带模型后缀。

## 配置文件

```text
codes/options/train/train_EDSR_d24_X{2,3,4}.yml
codes/options/test/test_EDSR_d24_X{2,3,4}.yml
codes/options/run/run_EDSR_d24_formal.yml
```

正式YAML规定每次训练200,000次更新、每5,000步完整DIV2K validation、每2,000步原子更新rolling resume。每个seed保存`best_val.pt`和`last.pt`；best只由DIV2K validation PSNR-Y选择。

## RTX 4060 pipeline smoke

```bash
python codes/run.py -opt codes/options/run/run_EDSR_d24_smoke_4060.yml
```

它只运行×4、seed0、10次更新，并在一张recovery数据上走通验证和通用测试。它不打开Set5/Set14/BSD100/Urban100/Manga109，也不产生正式质量结论。

## Featurize正式入口

```bash
bash scripts/cloud/run_featurize.sh \
  -opt codes/options/run/run_EDSR_d24_formal.yml
```

包装脚本负责RTX 4090/仓库/数据/checkpoint检查、执行通用plan、保存状态、生成带内部文件SHA-256的独立tar，并在归档验证后调用`featurize instance release`。正式训练须在4060 smoke与最终review之后另行promotion；准备好脚本不等于已经授权启动1,800,000次更新。

## 实验输出

```text
experiments/EDSR_d24_formal/
├── run_plan.resolved.yml
├── x4_seed0/
│   ├── train_config.resolved.yml
│   ├── test_config.resolved.yml
│   ├── train.log
│   ├── train_trace.jsonl
│   ├── models/{best_val.pt,last.pt}
│   ├── training_state/resume.pt
│   ├── val/{history.jsonl,step_*.jsonl}
│   └── test/{summary.csv,summary.json,test.log,<dataset>/per_image.*}
└── aggregate/{test_summary.csv,test_summary.json}
```

`experiments/`和`results/`均为运行产物目录，不进入Git。代码、YAML模板、`.env.example`和维护文档进入Git。
