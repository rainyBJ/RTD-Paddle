## Evaluation Part

### 在终端中进行如下操作

#### 加载数据集

``` bash
unzip ~/data/data112050/归档.zip  -d ~/work/data/
cp ~/data/data112070/checkpoint_best_sum_ar.pdparams ~/work/data/checkpoint_best_sum_ar.pdparams
```

#### Evaluate

num_workers = 0 时正确

```  bash
cd ~/work/RTD_RePro/
python -m main --window_size 100 --batch_size 32 --lr 1e-4 --stage 3 --epochs 20 --num_queries 32 --point_prob_normalize --eval --resume ../data/checkpoint_best_sum_ar.pdparams --feature_path ../data/I3D_features --tem_path ../data/TEM_scores
```

num_workers = 2 时错误

```bash
cd ~/work/RTD_RePro/
python -m main --window_size 100 --batch_size 32 --lr 1e-4 --stage 3 --epochs 20 --num_queries 32 --point_prob_normalize --eval --resume ../data/checkpoint_best_sum_ar.pdparams --feature_path ../data/I3D_features --tem_path ../data/TEM_scores --num_workers 2
```

### 问题

会出现如下错误