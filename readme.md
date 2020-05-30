run 
python main.py --dataset toumingzhi --model myganomaly --load_final_weights --batchsize 30
to test the images in data/toumingzhi/test
main.py prints the detection result 'pre_labels' where 0 indicates normal and 1 indicates anomaly.
