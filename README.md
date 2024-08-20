# Grad-CAM pytorch code

ImageNet sample images : https://github.com/EliSchwartz/imagenet-sample-images  
Code reference : https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82. 
https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/activations_and_gradients.py. 





# 모델을 이해하는 가장 쉬운 방법 :  
## 위의 결과랑 같게 나오게 해보기 
### Forward 부터 model class에 정의되어 있는걸로 해보고 순차적으로 직접 코딩하기


model.head(
    model.forward_features(inputs, masks = None)['x_norm_clstoken']
    )


x1 = model.prepare_tokens_with_masks(inputs, masks = None)
print(x1.shape)
x2 = x1
for blk in model.blocks:
    x2 = blk(x2)
print(x2.shape)
x3 = model.norm(x2)
print(x3.shape)
x4 = x3[:, 0]
print(x4.shape)
x5 = model.head(x4)
print(x5.shape)
x5