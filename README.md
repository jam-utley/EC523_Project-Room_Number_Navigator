# EC523_Project-Room_Number_Navigator
This is a  project done for Boston University, EC523.

We are making a VLM real-time object tracker and analyzer. 
For faster real time processing, we are using yolov7 as a real-time object tracker that isolates the image and then sends the clipped into the VLM for analysis and location marking.

All files are self-contained jupiter notebooks for easy download and reproducability of the results.

Florence2:

/QLoRA_Training/QLora_Florence2 (1).ipynb -- attempted QLoRA training on Florence2

Finetuning_Florence2.ipynb --successful finetuning of FLorence2

Graphing_Florence2_TrainVal.ipynb -- Graphing results from finetuning Florence2

Quantizing_Florence2.ipynb -- Successful quantization, but loss after quantization was far too high to have been successful.


Yolo: 

yolov7_door_plaque_training.ipynb -- Contains the code used to train and evaluate our implementation of yolov7

Qwen2-VL:

Will be updated when files are uploaded to the repository.
