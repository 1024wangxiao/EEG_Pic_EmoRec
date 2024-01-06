import torch
from torchvision import transforms
from model.CV.model_train import EmotionCNN
# 定义测试数据转换
from PIL import Image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cnn_predicted(path):
    # 加载模型
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(r'D:\table\EEG_Pic_EmoRec\Data\model_params\best_emotion_model.pth',map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    model.eval()

    # 读取测试图片
    img_path = path
    img = Image.open(img_path)

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0) # 增加第0维,作为批量维度
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    # 类别名称
    class_names =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    probs = torch.nn.functional.softmax(outputs, dim=1)
    result_dict = {}
    for i in range(len(class_names)):
        result_dict[class_names[i]] = probs[0, i].item()


    return result_dict

if __name__=="__main__":
    path=r'D:\table\EEG_Pic_EmoRec\Data\19.png'
    result_dict=cnn_predicted(path)
    print(result_dict)