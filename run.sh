# First stage

nohup python -u main.py --window_size 100 --batch_size 32 --stage 1 --num_queries 32 --point_prob_normalize > log/train_step1.log 2>&1 &

# Second stage for relaxation mechanism

nohup python -u main.py --window_size 100 --batch_size 32 --lr 1e-5 --stage 2 --epochs 10 --lr_drop 5 --num_queries 32 --point_prob_normalize --load outputs/checkpoint_best_sum_ar.pth > log/train_step2.log 2>&1 &

# Third stage for completeness head

nohup python -u main.py --window_size 100 --batch_size 32 --lr 1e-4 --stage 3 --epochs 20 --num_queries 32 --point_prob_normalize --load outputs/checkpoint_best_sum_ar.pth > log/train_step3.log 2>&1 &