python mnist_exp.py --epoch 50 --save "clean_mlp" > mlp_log.txt

python linear_probe.py --weight "clean_mlp_final.pth"

python mnist_exp.py --noisy --epoch 50 --save "noisy_mlp" > mlp_log.txt

python linear_probe_noisy.py --weight "noisy_mlp_final.pth"

python mnist_exp.py --noisy --epoch 20 --save "clean2noisy_mlp" --weight "clean_mlp_30.pth"

python linear_probe_noisy.py --weight "clean2noisy_mlp_final.pth"