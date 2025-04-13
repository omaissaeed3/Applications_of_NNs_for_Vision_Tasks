import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Compute gradients for a given input image

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image_pil = Image.open("Puppy2.jpg").convert("RGB")
image = preprocess(image_pil)
image.requires_grad_()
image.requires_grad = True
output = model(image.unsqueeze(0))
class_idx = torch.argmax(output)
output[0, class_idx].backward()


# Visualize saliency map
saliency = image.grad.abs().squeeze().permute(1, 2, 0)
plt.imshow(saliency.numpy(), cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()