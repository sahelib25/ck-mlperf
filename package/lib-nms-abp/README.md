Once this package is added as a CK dependency to your object-detection program, you can simply call the NMS routines as 

## For SSD-ResNet34
```
  NMS_ABP<TOutput1DataType, TOutput2DataType, R34_Params> nms_abp_processor;
  R34_Params modelParams; //gives the model constants 
```
## For SSD-MobileNet
```
  NMS_ABP<TOutput1DataType, TOutput2DataType, MV1_Params> nms_abp_processor;
  MV1_Params modelParams; //gives the model constants
```
