#python -O quantjia.py train_model M1_T10_B256_C4_E300_S3200_D200 200
#python -O quantjia.py validate_model M1_T10_B256_C4_E300_S3200_D200 5
#python -O quantjia.py train_model M1_T10_B256_C4_E300_S3200
#python -O quantjia.py validate_model M1_T10_B256_C4_E300_S3200 5

python -O quantjia.py train_model M1_T20_B248_C31_E300_S3200_Little5 1980-10-16 2017-01-01 
python -O quantjia.py validate_model M1_T20_B248_C31_E300_S3200_Little5 2017-01-02 2017-04-01
#python -O quantjia.py predict_today M1_T20_B248_C4_E300_S3200_Little_D300

python -O quantjia.py train_model M1_T20_B248_C31_E300_S3200_Little_D300 2016-01-01 2017-01-01
python -O quantjia.py validate_model M1_T20_B248_C31_E300_S3200_Little_D300 2017-01-02 2017-04-01
#python -O quantjia.py predict_today M1_T10_B248_C3_E300_S3200_Little
#python -O quantjia.py train_model M1_T10_B248_C4_E300_S3200_Little 
#python -O quantjia.py validate_model M1_T5_B248_C4_E300_S3100_Little 5  


#python -O quantjia.py train_model M1_T20_B248_C4_E300_S3200_Little 1980-10-16 2017-01-01
#python -O quantjia.py validate_model M1_T20_B248_C4_E300_S3200_Little 2017-01-02
#python -O quantjia.py predict_today M1_T5_B248_C3_E300_S3200_Little 

#python -O quantjia.py train_model M1_T30_B248_C3_E300_S3200_Little_D300
#python -O quantjia.py validate_model M1_T30_B248_C3_E300_S3200_Little_D300 5
#python -O quantjia.py predict_today M1_T30_B248_C3_E300_S3200_Little_d300 
#python -O quantjia.py train_model M1_T10_B248_C3_E300_S3200_Little 
#python -O quantjia.py validate_model M1_T10_B248_C3_E300_S3200_Little 5
