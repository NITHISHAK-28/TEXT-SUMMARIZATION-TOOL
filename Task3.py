import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Load images and resize
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    if max_size:
        size = max(image.size)
        if size > max_size:
            image = image.resize((max_size, max_size), Image.ANTIALIAS)
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    transform = transforms.ToTensor()
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# Display image
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Define Content & Style Loss
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                  '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Load model and images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features.to(device).eval()

content = load_image("content.jpg")
style = load_image("style.jpg", shape=[content.size(2), content.size(3)])
target = content.clone().requires_grad_(True)

# Define optimizer
optimizer = optim.Adam([target], lr=0.003)

# Define style & content features
style_features = get_features(style, vgg)
content_features = get_features(content, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Style weights
style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5,
                 'conv4_1': 0.3, 'conv5_1': 0.1}
content_weight = 1e4
style_weight = 1e2

# Training
steps = 300
for step in range(steps):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.4f}")

# Show result
imshow(target, title="Styled Image")
