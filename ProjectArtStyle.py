import os
import streamlit as st
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI  # Uncomment if using LLM for prompt engineering
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from PIL import Image
import io
import requests
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch import nn
from google.colab import files

# Set Streamlit page config
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up LLM (if needed for prompt engineering)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)

# --- Session State for Version History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- File Upload ---
st.title("Art Style Transfer (LLM-powered)")

uploaded_file = st.file_uploader("Upload an image (jpg/png):", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
else:
    image = None

# --- Style Selection ---
STYLES = [
    "Van Gogh (Starry Night)",
    "Monet (Water Lilies)",
    "Picasso (Cubism)",
    "Upload your own style image"
]

style = st.selectbox("Choose an art style:", STYLES)

# Built-in style images (small demo images)
BUILTIN_STYLE_IMAGES = {
    "Van Gogh (Starry Night)": "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg",
    "Monet (Water Lilies)": "https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg",
    "Picasso (Cubism)": "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg"
}

style_image = None
if style == "Upload your own style image":
    uploaded_style = st.file_uploader("Upload a style image (jpg/png):", type=["jpg", "jpeg", "png"], key="style")
    if uploaded_style:
        style_image = Image.open(uploaded_style).convert("RGB")
else:
    response = requests.get(BUILTIN_STYLE_IMAGES[style])
    style_image = Image.open(io.BytesIO(response.content)).convert("RGB")

# --- Style Transfer Button ---
col1, col2 = st.columns([2, 1])
with col1:
    transfer_btn = st.button("Apply Style Transfer")
with col2:
    revert_btn = st.button("Revert to Previous")

# --- Style Transfer Parameters ---
st.sidebar.header("Style Transfer Settings")
imsize = st.sidebar.slider("Image Size (px)", min_value=256, max_value=800, value=512, step=64)
num_steps = st.sidebar.slider("Optimization Steps", min_value=100, max_value=1000, value=400, step=50)
style_weight = st.sidebar.number_input("Style Weight", min_value=1e3, max_value=1e8, value=1e6, step=1e3, format="%.0f")
content_weight = st.sidebar.number_input("Content Weight", min_value=1.0, max_value=1e3, value=100.0, step=1.0, format="%.0f")

# --- Device Info and Warning ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write("PyTorch device:", device)
st.sidebar.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.sidebar.write("CUDA device:", torch.cuda.get_device_name(0))
else:
    st.sidebar.warning("⚠️ GPU (CUDA) is NOT available. Style transfer will run on CPU and be much slower.\nTo use your GPU, install the CUDA-enabled version of PyTorch and ensure your drivers are up to date.")

# --- Helper functions for neural style transfer ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def image_loader(image, imsize=256):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    return image

# --- Display the Style Image ---
if style_image:
    st.image(style_image, caption="Style Image", use_column_width=True)

# --- Robust Neural Style Transfer Implementation (based on PyTorch tutorial) ---
import copy

def run_style_transfer(content_img, style_img, imsize=512, num_steps=400, style_weight=1e6, content_weight=100):
    # Use the global device variable
    global device
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    content_img = loader(content_img).unsqueeze(0).to(device, torch.float)
    style_img = loader(style_img).unsqueeze(0).to(device, torch.float)
    input_img = content_img.clone()
    cnn = vgg19(pretrained=True).features.to(device).eval()
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean.clone().detach().view(-1, 1, 1)
            self.std = std.clone().detach().view(-1, 1, 1)
        def forward(self, img):
            return (img - self.mean) / self.std
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_losses = []
    style_losses = []
    model = nn.Sequential(Normalization(normalization_mean, normalization_std))
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]
    input_img = input_img.requires_grad_(True)
    optimizer = torch.optim.LBFGS([input_img])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
            return loss
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    image = input_img.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# --- Main Logic ---
if transfer_btn and image and style_image:
    with st.spinner("Applying style transfer (this may take a minute)..."):
        stylized_image = run_style_transfer(image, style_image, imsize=imsize, num_steps=num_steps, style_weight=style_weight, content_weight=content_weight)
        st.session_state.history.append({
            "original": image,
            "stylized": stylized_image,
            "style": style,
        })

# --- Revert to Previous ---
if revert_btn and len(st.session_state.history) > 1:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    image = prev["original"]
    stylized_image = prev["stylized"]
    style_prompt = prev["style"]
    st.success("Reverted to previous version.")
elif revert_btn:
    st.warning("No previous version to revert to.")

# --- Show Side-by-Side Comparison ---
if st.session_state.history:
    last = st.session_state.history[-1]
    orig = last["original"]
    stylized = last["stylized"]
    style_used = last["style"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Image**")
        st.image(orig, use_column_width=True)
    with c2:
        st.markdown(f"**Stylized Image ({style_used})**")
        st.image(stylized, use_column_width=True)
    # --- Download Stylized Image ---
    buf = io.BytesIO()
    stylized.save(buf, format="PNG")
    st.download_button(
        label="Download Stylized Image",
        data=buf.getvalue(),
        file_name="stylized_image.png",
        mime="image/png"
    )

uploaded = files.upload() 