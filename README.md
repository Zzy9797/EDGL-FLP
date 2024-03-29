# EDGL-FLP
Enhanced Discriminative Global-Local Feature Learning with Priority for Facial Expression Recognition

## Requirements
- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0
- torchsampler

## Training

- Step 1: prepare the datasets, and make sure they have the structures like following (take RAF-DB as an example):
 
```
./RAF-DB/
         train/
               0/
                 train_09748.jpg
                 ...
                 train_12271.jpg
               1/
               ...
               6/
         test/
              0/
              ...
              6/

[Note] 0: Neutral; 1: Happiness; 2: Sadness; 3: Surprise; 4: Fear; 5: Disgust; 6: Anger
```

- Step 2: download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1WgyDpDtiDBuESME1GIByeQmJ0-PukzM8/view?usp=share_link) and put it to ***./checkpoint***.
    
- Step 3: change ***data_path*** in *main.py* to your path 

- Step 4: run ```python main.py ```


## Note
- When training from scratch or pre-training, use *main.py* as well.


**Acknowledgments**

Our work is based on the following work, thanks for the code:

https://github.com/zengqunzhao/MA-Net
