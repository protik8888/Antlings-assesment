Task 1:
The dataset is VisDrone which is for object detection and tracking in drone based aerial image
The dataset contains the following class distribution:
  pedestrian: 13656
  people: 3776
  bicycle: 847
  car: 17371
  van: 3471
  truck: 1665
  tricycle: 300
  awning-tricycle: 390
  bus: 1753
  motor: 3445
The dataset was structured in the standard YOLO PyTorch format. Every image in the images/ folder has a corresponding .txt file in the labels folder containing the bounding box annotations (class_id x_center y_center width height).

The dataset is divided into train, test and validation

The dataset contains 10 classes. Among those 3 classes are cars, people and pedestrians . People and pedestrians are actually human. Pedestrian (0), people(1), cars(3)


Augmentation and Pre processing: Some augmentation and pre processing techniques were applied on the dataset such as horizontal rotation and flipping, brightness adjustment. Also while training the dataset the resolution of the images were set to 640*640 pixels. Some augmentation and pre processing were handled automatically by Ultralytics such as scaling,normalization and mosaic augmentation.

challenges noticed in the dataset : 
1.The dataset is imbalanced. Compared to cars the numbers in other classes are very small. In the dataset there are 42% of cars where pedestrians and people are very low, only 23% and 7.9%. This imbalance creates biases towards majority classes.


2.The images are from high altitude and some objects such as people and pedestrian images are in low pixels. It means that those bounding boxes have very low pixel density. It is hard to extract meaningful features.

The annotations are quite problematic too; some annotations overlap on each other completely. Sometimes pedestrians and people are annotated with mistakes. Here are some images:













Task-02: Model Training: For the object detection framework, I selected YOLOv8 (yolov8n.pt Nano architecture).The ultimate goal of this pipeline is to track objects in a high-resolution, 60fps drone video. While heavier models might offer slight accuracy increase, they show massive inference latency. YOLOv8n provides the optimal real-time inference speed and detection accuracy.

Instead of training from scratch, I utilized fine-tuning. The model was initialized with pre-trained COCO weights, allowing it to utilize pre-trained low-level features (like edges and shapes) and significantly reducing the time required to train the dataset.

Hyperparameters : Epoch: 30,
                                 Image size :640*640 pixels
                                 Batch size :16

Results and evaluations: During the validation phase, the model exhibited distinct performance variations across the target classes (Pedestrian, People, and Car). 
Cars were the dominant class.

Here is the class wise result:

As per the result cars shows mAP50 of 0.709 (70.9%) and recall of 0.749. It is very normal as cars have so many images and annotations. But the pedestrian and people class struggles very much. Pedestrians have a mAP50 of 0.288(28.8%) and recall of 0.338. And in the case of people it is 0.221 and 0.22. Which shows that these two human classes really struggle . It is because of the lower number of annotations and low quality of the images.


                                                               Confusing matrix



Task-03: Human & Car Detection with Human Counting :
 human_count = 0 (pedestrian and people are counted together as human)
 car_count = 0
for box in results[0].boxes:
    class_id = int(box.cls[0])
if class_id in [0, 1]:          
 (if it is detected that the class is pedestrian 0 and people 1 then together they are human)
        human_count += 1
    elif class_id == 3:
        car_count += 1




Task-04: Object Tracking: part 1 ByteTrack:
For this part ByteTrack was implemented. While tracking bytetrack was implemented because of its two stage association method:
 
High-Confidence Matching: bytetrack matches the high confidence yolo detections to existing trajectories.

Low-Confidence Recovery: bytetrack does not delete the low confidence matching instead it tries to match with the existing tracking.

print("Starting ByteTrack Tracker...")
results = model.track(
    source=source_path,
    conf=0.25,
    classes=[0, 1, 3],
    tracker="bytetrack.yaml",
    save=True,
    project='Antlings_Assessment',
    name='tracking_output')
Setting conf=0.25 means I am  telling the model: If it is less than 25% sure that this object is a pedestrian, person, or car, ignore it completely. 

Tracking video is added as ByteTrackingsample.avi

--- FINAL TRACKING RESULTS ---
Total UNIQUE Humans Tracked: 7
Total UNIQUE Cars Tracked: 2



Part 2 Bot-Sort: BoT-SORT (Robust Tracking with Camera Motion Compensation) was implemented to resolve the ID switching caused by the drone's movement
BoT-SORT builds upon ByteTrack's low-confidence association logic but introduces two critical mathematical upgrades that make it ideal for drone footage:
Camera Motion Compensation (CMC): Unlike a stationary traffic camera, a drone is constantly moving. When the drone moves left, all objects mathematically appear to move right. BoT-SORT uses background feature matching to calculate the camera's exact movement frame-by-frame. It then subtracts this  from the equation before applying the Kalman filter. This allows the tracker to accurately predict where a car or pedestrian is going, regardless of how erratically the drone flies.
Visual Appearance Matching (ReID): In heavy urban traffic, bounding boxes frequently overlap, or targets pass behind obstacles (like a car driving under a bridge). BoT-SORT extracts visual features (like the specific color of a car or a pedestrian's shirt) to maintain ID persistence even when spatial math fails.
--- FINAL BoT-SORT TRACKING RESULTS ---
Total UNIQUE Humans Tracked: 8
Total UNIQUE Cars Tracked: 2
Tracking video is added as Bot Sortsample.avi


Another set of video footage was tracked during the comparison
ByteTrack: video: ByteTrack60fps.avi

--- FINAL BYTETRACK RESULTS ---
Total UNIQUE Humans Tracked: 34
Total UNIQUE Cars Tracked: 230

Bot Sort: video: Bot Sort60fps.avi

--- FINAL BoT-SORT RESULTS ---
Total UNIQUE Humans Tracked: 34
Total UNIQUE Cars Tracked: 210
 
Comparison: Although the both tracking may seem same their underlying workings are different.
ByteTrack does not have camera motion compensation means a moving camera can not track objects and does not track how the object looks like. Bot Sort have the underlying workings of ByteTrack like low confidence recovery but adds camera motion compensation and can extract visual colors and appearances. But ByteTrack is faster than Bot Sort. So it seems that  BoT-SORT is the ideal algorithm for general drone deployment.



Task-05: Evaluation & Visualization : 
Prediction outputs: The training achieved the mAP50 of 0.288 and 0.211 for pedestrians and people and the recall was 0.338 and 0.22. For the car class the mAP50 is 0.709 and the recall was 0.749.

Confusion matrix:



Rows = what the model predicted
Columns = what the actual label is
Diagonal (dark blue) = correct predictions
Off-diagonal = mistakes
The confusion matrix reveals that the model performs best on larger objects such as cars with 9956 correct detections.but struggles with small objects like pedestrians and people, where a significant number are misclassified as background 6212 and 3911 respectively. This is
consistent with the known challenge of small object detection in aerial imagery. Rare classes  such as bus and awning-tricycle show poor performance due to class imbalance in the training data.










Counting visualization: for counting people and pedestrians were merged as human class




strengths : The main strength of the dataset is that it has a huge amount of images. It has great potential to create a detection pipeline for a drone to work correctly. Tracking also works quite nice. High altitude views of the images are great to train the dataset for this work too.

limitations : The limitations are significant in this dataset. As we can see, the annotations of cars are huge compared to people and pedestrians. Also some annotations are poorly made, like pedestrians are made people and vice versa. Annotating two people riding a car is also very problematic.

challenges faced: Working with the huge dataset is indeed a challenge. It takes time and resources to train these images. Moreover, the dataset is very imbalanced. It  is a challenge to create a meaningful model out of it. Balancing the dataset is a hassle. To overcome the class imbalance we could have used some pipelines like merging the people and pedestrians as human in the training, it may have improved a bit but not much. The best way would be to increase the people and pedestrian classes and correct the annotations from scratch.


Object tracking: while tracking the objects specially people, pedestrians and cars we have to think from the perspective of a drone. Comparing ByteTrack and Bot Sort we can see that Bot Sort is more reliable because it can track colors and other features but not very useful in low end hardware as it needs more computing power.    https://drive.google.com/drive/folders/15gFJva50_npRsZpTnflLWswWyNQXLYlb?usp=sharing








