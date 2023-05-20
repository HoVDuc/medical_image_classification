### medical_image_classification

### Download data

```bash
gdown 1Q0Oah_S1DZflHRv3RUJyoJG_Hm5LNXCP
```

### Data Overview


### Kết quả training
|Backbone   |Sampling            | F1-Scores     |  mAP| Precision| Recall|
|-----------|:------------------:|:-------------:|-----|----------|-------|
|Resnet18   |None                |0.0            |  0.0|       0.0|    0.0|
|Resnet50   |None                |0.0            |  0.0|          |       |
|Resnet50   |Undersampling       |0.348          |0.381|     0.358|  0.346|
|Resnet50   |Focal loss          |0.643          |0.632|     0.652|  0.639|
|ECA-NFNet  |None                |0.664          |NAN  |          |       |
|ECA-NFNet  |Undersampling       |0.625          |0.526|     0.609|  0.629|
|ECA-NFNet  |Focal Loss          |0.665          |0.628|     0.672|  0.664|

