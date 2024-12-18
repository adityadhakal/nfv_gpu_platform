import torch
import torchvision
import time
import pretrainedmodels
from torch.autograd import Variable

model = None
image = None
stream = None
#loads the model
def load_model_to_gpu():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.cuda()
    model.eval()

#creates a tensor in GPU
def create_gpu_tensor():
    image = torch.rand(1,3,224,224)
    image.cuda()
    data_ptr = image.data_ptr()
    return data_ptr

def create_stream():
    stream = torch.cuda.Stream()
    return stream.cuda_stream

#executes the image
def execute_image():
    output = model(image)


'''



bigger_image = [Variable(torch.rand(1,3,300,400).cuda()),Variable(torch.rand(1,3,500,400))]
#bigger_image = bigger_image.cuda()

#model_name = 'densenet201'
#model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')



#model = model.cuda()
model.eval()

#print("Model input size ")
#print(model.input_size())

#first execution
print(model(torch.rand(1,3,800,800)))
#model(example)

#2nd time
t = time.process_time()
#output = model(bigger_image)
#output = model(example)
torch.cuda.synchronize()
elapsed_time = time.process_time()-t

print("Elapsed time in seconds ")
print(elapsed_time)


# features of the model
print("Input size of the model ")
#print(model.input_size())

#output = model(example)
#print(output)
#model(example)


#2nd time
t = time.process_time()

#output = model(bigger_image)
torch.cuda.synchronize()
elapsed_time = time.process_time()-t

#print (output)

print (elapsed_time)
#let's serialize the models

traced_script_module = torch.jit.trace(model,[[torch.rand(3,300,400)], [torch.rand(3,500,400)]])
traced_script_module.save("torch_fast_rcnn.pt")

'''
