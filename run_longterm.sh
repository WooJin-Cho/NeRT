for data_name in national_illness
do
for model_name in nert_multi
do
for train_rat in 0.7
do
for valid_rat in 0.15
do
for lr in 1e-3
do
for max_scale in 1
do
for hidden_dim in 100
do
for enc_dim in 10
do
for sine_dim in 50
do
for learn_freq in 5.
do
for inner_freq in 1.
do
for sine_emb_dim in 10
do
for time_emb_dim in 50
do
for f_emb_dim in 50
do
for seed in 0
do



python -u train_longterm.py --data_name $data_name --model_name $model_name \
--train_rat $train_rat --valid_rat $valid_rat --lr $lr --max_scale $max_scale --hidden_dim $hidden_dim \
--enc_dim $enc_dim --sine_dim $sine_dim --learn_freq $learn_freq --inner_freq $inner_freq \
--sine_emb_dim $sine_emb_dim  --time_emb_dim $time_emb_dim --f_emb_dim $f_emb_dim --seed $seed \
> ./longterm_results.csv

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
done
done