import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Load content and style images
content_image = load_image("./Puppy2.jpg")
style_image = load_image("./Puppy2.jpg")  # You can replace with a different style image

# Clone content image to initialize the target image
image = content_image.clone().detach().requires_grad_(True)


vgg = models.vgg19(pretrained=True).features[:22].eval()
for param in vgg.parameters():
    param.requires_grad = False


def content_loss(target, content):
    return torch.mean((target - content) ** 2)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

def style_loss(target, style):
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)


with torch.no_grad():
    content_features = vgg(content_image)
    style_features = vgg(style_image)


optimizer = optim.Adam([image], lr=0.01)
for i in range(500):
    target_features = vgg(image)  # <-- You must optimize "image", not "content_image"
    
    c_loss = content_loss(target_features, content_features)
    s_loss = style_loss(target_features, style_features)
    loss = c_loss + 1e6 * s_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")


output = image.detach().squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
plt.imshow(output)
plt.title("Stylized Image")
plt.axis("off")
plt.show()
