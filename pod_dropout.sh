for TASK in cola
do
for LORA in "" "--lora"
do
for D in 0.05 0.1 0.15 0.2 # 0.0 already done
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --lr $LR $LORA --project dropout --dropout $D"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --lr $LR $LORA --project dropout --dropout $D > logs/dropout_${TASK}_${LORA}_d${D}.log 2>&1
done
done
done