
for data_name in depts_traffic
do
for model_name in nert_uni
do
for block_num in 1
do
for lr in 1e-3
do
for max_scale in 1
do
for hidden_dim in 50
do
for enc_dim in 10
do
for sine_dim in 10
do
for learn_freq in 10.
do
for inner_freq in 1.
do
for sine_emb_dim in 10
do
for time_emb_dim in 30
do
for seed in 0
do


python -u train_periodic.py --data_name $data_name --model_name $model_name \
--block_num $block_num --lr $lr --max_scale $max_scale --hidden_dim $hidden_dim \
--enc_dim $enc_dim --sine_dim $sine_dim --learn_freq $learn_freq --inner_freq $inner_freq \
--sine_emb_dim $sine_emb_dim  --time_emb_dim $time_emb_dim --seed $seed \
> ./periodic_results.csv



done
done
done
done
done
done
done
done
done
done
done
done
done