##############################################################

# - - - - - - - - - - - - - - - - - - - - - # 
#                   LA
# - - - - - - - - - - - - - - - - - - - - - # 

expname="LA_runs"
version="1"
numlb=8 # 4, 8, 16
gpuid=0

python3 ./code/test_performance_3d.py \
    --root_path ./data/LA/ \
    --res_path ./results/LA/ \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --flag_check_with_best_stu \
    --labeled_num ${numlb}
