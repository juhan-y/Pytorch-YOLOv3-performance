# Pytorch-YOLOv3-performance

## performance explanation of code in korean

2번에서의 YOLOv3-darknet으로 roboflow에서 가져온 dataset으로 학습시켜서 model을 만들어보았다. 작업은 colaboratory에서 진행했고 리눅스명령어를 사용해서 학습을 진행했다.
roboflow에서 가져온 dataset은 image가 총 627장이다.

![image](https://user-images.githubusercontent.com/81463668/113807586-63822580-979f-11eb-8bc8-40a747aa8fbf.png)

구글드라이브를 마운트에서 구글 드라이브에 갖가지 library들이나 yolov3를 저장하고 꺼내서 쓸 용도이다. dataset도 구글 드라이브에 있으므로 가져와서 쓸 예정이다.

구글 드라이브 내부에 github를 clone해서 YOLOv3를 가져와 저장한다.

![image](https://user-images.githubusercontent.com/81463668/113807611-6d0b8d80-979f-11eb-89e9-6ecc51bef849.png)
![image](https://user-images.githubusercontent.com/81463668/113807615-6ed55100-979f-11eb-8de5-e94a29036466.png)
![image](https://user-images.githubusercontent.com/81463668/113807626-739a0500-979f-11eb-8921-427c20fb498f.png)
requirements.txt를 실행해서 YOLOv3구현에 필요한 library들을 한번에 import!

![image](https://user-images.githubusercontent.com/81463668/113807635-785eb900-979f-11eb-9416-9c947a346dc7.png)

weights 폴더로 이동한 뒤 bash명령어를 통해 pretrain 된 가중치를 가져온다.
weight를 초기화하고 학습전에 넣어서 좀 더 학습이 빠르게 진행될 수 있다.

![image](https://user-images.githubusercontent.com/81463668/113807640-7d236d00-979f-11eb-86b6-0151988dd525.png)

구글 드라이브에 미리 업로드한 dataset을 처리해야하므로 data폴더로 이동

![image](https://user-images.githubusercontent.com/81463668/113807651-80b6f400-979f-11eb-8575-dc0e9ac89ce5.png)

업로드된 dataset파일들을 image와 label로 나누어서 unzip을 진행한다.

![image](https://user-images.githubusercontent.com/81463668/113807655-844a7b00-979f-11eb-924d-513d3fa3a4ba.png)

결과창에서는 unzip파일을 찾을 수 없다고 나오는데 이는 이미 진행 후 zip파일들을 삭제했기 때문이고 이미 잘 unzip이 되었음을 알 수 있다.

![image](https://user-images.githubusercontent.com/81463668/113807660-87456b80-979f-11eb-815e-ea2752df6d32.png)

이것은 label폴더안에 zip파일들을 unzip해주는 것인데 만약 zip파일이 존재해서 제대로 unzip이 이루어졌다면 위 상태창과 같이 경로가 이동했다고만 나올 것.

![image](https://user-images.githubusercontent.com/81463668/113807675-8d3b4c80-979f-11eb-890c-fea8618dce15.png)

이제 dataset을 train과 valid set으로 분할하고 train set과 valid set 내부의 데이터를 하나의 리스트안에 경로저장을 하게된다.
나중에 이 경로를 가져와서 training을 진행하게 된다.

![image](https://user-images.githubusercontent.com/81463668/113807686-92000080-979f-11eb-9104-dcb45c5fe3bb.png)

만약 unzip을 한 뒤에 zip파일들을 지워주지않고 이 ipynb파일을 한번 더 돌리게 되면 위의 unzip코드가 여러번 실행이되므로 같은 dataset이 여러번 unzip되고 폴더안에 같은 사진들이 중복되어 생성될 것이다. 그러므로 zip파일을 지워줘야 한다.
상태창들은 이미 zip파일 제거를 실행하고 한번 더 실행한 결과이기 때문에 zip파일을 찾을 수 없다고 나온다. 이미 진행한 코드이므로 zip파일은 삭제되었다.

![image](https://user-images.githubusercontent.com/81463668/113807695-95938780-979f-11eb-8011-c96edd507759.png)

이제 train을 시키기전에 내가 가져온 이 YOLOv3는 COCO dataset에 맞춰져있어서 config파일이나 설정 파일들이 COCO dataset설정으로 되어있다. 그렇기 때문에 여러 config파일들을 수정했는데 이 코드부분도 COCO dataset과는 달리 class의 개수가 5개인 dataset을 사용하므로 class의 개수를 5개로 수정해준 것이다. 이외에 다른 config파일들을 약간 수정하였는데 config파일들은 수정한 코드부분을 제외한 대부분의 코드는 건드리지 않았으므로 설명하지 않고 넘어가도록 하겠다. 만약 궁금하다면 config폴더 내부에 들어가서 각 hyperparameter들을 확인하면 되겠다.

![image](https://user-images.githubusercontent.com/81463668/113807700-99bfa500-979f-11eb-8db3-71b2710f5e70.png)

이제 train을 시킨다. train.py를 불러와서 python가상환경에서 parser의 명령어를 이용해서 설정해준 값으로 train을 시작한다. --pretrained_weights checkpoints~ 이부분은 원래 위의 코드에서 가져온 pretrained weights를 넣어주는데 지금 이 코드는 이미 epoch을 300번 진행하고 google colab의 runtime이 끊겨서 학습중간에 저장된 299번째 pth(300 epoch일때의 path)파일을 가져와서 그 부분부터 다시 시작을 하는 과정이다. epoch은 2000을 설정해뒀지만 개발자옵션에서 명령어를 넣어 runtime이 끊기지 않도록 trick을 썼음에도 불구하고 google colab환경이 불안정해서 일단은 계속적으로 학습이 되도록 하였다.
학습이 중간에 끊기더라도 저장된 parameter들로 다시 학습을 시키도록 한 것이다.
추가적으로 validation set은 training 중간마다 넣어서 training셋이 얼마나 성능이 나오는지 판단하도록 쓰인다.

![image](https://user-images.githubusercontent.com/81463668/113807718-a04e1c80-979f-11eb-80d1-ed11fd64b0cb.png)

이제 training을 시켰기 때문에 그 완성된 model을 시각적으로 정량적인 지표를 확인할 것이다. tensorboard를 가져오기위해 tensorflow를 가져오고 tensorboard에 학습될 때 저장되었던 log들을 업로드해서 그래프로 나타내려고 한다.


## 결과분석

빨간색의 그래프는 epoch을 300번 돌렸을 때 결과이다.

![image](https://user-images.githubusercontent.com/81463668/113807745-acd27500-979f-11eb-9cfa-8a2bd80dfcad.png)

train set에 대한 model 성능을 나타낸 그래프이다.
학습이 진행됨에 따라 class loss, iou loss, loss, obj loss 모두 줄어듬을 확인할 수 있고
class loss는 그 클래스에 해당하는 예측값과 ground-truth값의 차이인데 점점 줄어들고 있는 것을 보아 잘 학습이 되고 있는 것을 알 수 있고 중간에 overshoot한 부분이 있는데 이부분은 image를 가져와서 직접 눈으로 봐야할 것 같지만 아마도 특정 class에서 인식을 잘하지 못하는 경우 인것같다. 그로 인해 왼쪽 아래에 있는 loss값에도 overshoot이 있는 것으로 판단된다.
iou loss는 ground-truth값과 bounding box의 정보가 일치하지 않는 정도라고 할 수 있는데 점점 줄어드는 것으로 보아 bounding box가 객체에 점점 맞게 생성되는 것이라고 할 수 있겠다. obj loss는 class loss와 비슷하게 bounding box안에 객체가 있을 확률과 ground-truth값을 비교해서 나온 값인데 이 부분도 감소하고 있다.

![image](https://user-images.githubusercontent.com/81463668/113807763-b22fbf80-979f-11eb-8744-5275cd990dbf.png)

이제 train한 model에 대해서 valid set에 적용한 결과를 보자. 
F1 score는 Precision과 Recall의 조화평균인데 점점 증가하고 있는 모습을 보인다.
precision과 recall도 학습이 진행되면서 점차 증가하는 추세를 보인다.
mAP로 마찬기지로 증가하고 있는데 사실 좋은 결과라고 보기는 어렵다.
왜 이렇게 판단했냐면 Yolo논문에 따르면 정확도가 R-CNN기법에 비해서 낮다고는 하지만 mAP가 0.5~0.75정도는 나왔기 때문이다. 그래서 학습을 좀 더 진행하면 더 좋은 성능의 model이 될 수 있지 않을까히고 좀 더 training을 시켜보았다.

![image](https://user-images.githubusercontent.com/81463668/113807780-b78d0a00-979f-11eb-96e4-f6ae657dc019.png)

이제 그래프에서 파란색 선부분이 빨간색 선끝부분에서 연장되었다고 보면 되는데 빨간색 선에 해당하는 학습이 종료되고 그 마지막의 parameter를 적용해서 학습을 다시 시킨 것이다. 그러므로  간색 300epoch, 파란색 약 250epoch으로 학습을 진행시켜 총 550epoch으로 모델을 반복적으로 학습시킨 셈이다. 이제 그래프를 보자면 생각보다 성능이 나아지지 않았다. 전제적으로 빨간색 선의 끝부분과 비교해보면 250의 epoch이 진행되면서 f1, precision, recall이 모두 향상되지 않았으명 마찬가지로 mAP도 0.25정도 부근에서 더 이상 좋아지지 않는 모습을 보였다.

왜 학습을 더 시켜도 더 이상 model의 성능이 좋아지지 않을까에 대한 분석을 해보았다.
가장 주된 이유는 dataset자체가 627개라는 적은 개수의 image로 학습을 시키기 때문이다. 아무래도 data augmentation이나 데이터개수 자체를 늘려서 학습을 하게되면 더 좋은 성능을 기대할 수 있을 것 같다. 
추가적으로 적은 개수의 image로 비교적 많은 epoch으로 학습을 진행하다보니 overfitting의 문제가 있는 것같다. 물론 이 프로젝트는 residual 층이나 dropout, momemtum, adam optimizer 기법으로 overfitting문제를 그나마 피하고는 있는 것 같다. 적은 수의 image를 가지고 학습을 하기에 epoch을 증가시켜도 더 이상의 모델의 향상이 이루어지지 않는 것으로 보인다. 


### 실행하기위해 만든 Colab주소:
### https://colab.research.google.com/drive/1tFRKda4d6vKD6geag0WNzkU8y5DA8X9U?usp=sharing 
### github 주소: https://github.com/juhan-y/PyTorch-YOLOv3





