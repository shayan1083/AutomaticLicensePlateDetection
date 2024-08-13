import cv2
import torchvision
from torchvision.transforms import v2
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from skimage import io

from flask import Flask, jsonify

app = Flask(__name__)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

data_dir = '../images/bilinear_images_5x500'
torch.manual_seed(111)
transform = v2.Compose([
    v2.Grayscale(),
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32), # normalizes to range [0,1]
])
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

class net_4(nn.Module):
    def __init__(self):
        super(net_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64,128, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*2*2, 64)
        self.fc2 = nn.Linear(64, 128) 
        self.fc3 = nn.Linear(128,36) # 36 classes for 10 digits + 26 letters        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        #print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

def get_detections(img,net): # convert image to yolo format, predict using yolo model
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image,detections

def non_maximum_supression(input_image,detections): # filter detections based on confidence and probability score
    boxes=[]
    confidences=[]
    image_w,image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4: 
            class_score = row[5] # probability score of license plate
            if class_score > 0.25: 
                cx,cy,w,h = row[0:4] # extract bounding box coordinates
                # scale boudning box coordinates to original size
                left = int((cx-0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                confidences.append(confidence)
                boxes.append(box)
    # convert lists to required format for NMS
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # Non Maximum Suppression (NMS)
    # removes redundant overlapping boxes based on their confidence scores
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    return boxes_np,confidences_np,index

def drawings(image, boxes_np,confidences_np,index):
    cropped_images = []
    bboxes = []
    # iterate over each index provided by NMS
    for ind in index:
        x,y,w,h = boxes_np[ind] # extract bounding box coordinates
        bb_conf = confidences_np[ind] # extract confidence
        conf_text = 'plate {:.0f}%'.format(bb_conf*100) # format confidence text
        roi = image[y:y+h, x:x+w]
        cropped_images.append(roi)
        bboxes.append((x, y, w, h))
    return image, cropped_images, bboxes

def bbox_predictions(img,net):
    input_image,detections=get_detections(img,net)
    boxes_np,confidences_np,index = non_maximum_supression(input_image,detections)
    result_img, cropped_images, bboxes = drawings(input_image,boxes_np,confidences_np,index)
    return result_img, cropped_images, bboxes

def preprocess_image(image):
    image = cv2.resize(image, (300,150))
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # convert grayscale image to binary image (black and white)
    _, binary = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def find_contours(binary_image):
    # detect contours, which are curves joining all the continuous points along a boundary
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    character_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 53 <= h <= 81 and 19 <= w <= 46:
            character_contours.append(contour) 
    return character_contours

def extract_characters(contours, binary_image):
    character_images = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key = lambda x: x[0])
    for box in bounding_boxes:
        x, y, w, h = box
        char_image = binary_image[y:y+h, x:x+w]
        character_images.append(char_image)
    return character_images

def segment_characters(license_image):
    binary_image = preprocess_image(license_image)
    contours = find_contours(binary_image)
    character_images = extract_characters(contours,binary_image)
    return character_images

def recognize_character(img, model, class_names, transform):
    model.eval()
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probabilities = output.squeeze().cpu().numpy()
    predicted_index = probabilities.argmax()
    predicted_class = class_names[predicted_index]
    confidence_score = probabilities[predicted_index]
    confidence_scores = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    return predicted_class, confidence_score, confidence_scores

def full_code(path, detect_model, read_model, class_names):
    full_image = io.imread(path)
    _, cropped_images, _ = bbox_predictions(full_image,detect_model) # detect license plates in image
    license_plate_image = cropped_images[0] # get first license plate detected
    char_imgs = segment_characters(license_plate_image) # segment characters on license plate
    transform = v2.Compose([
        v2.Grayscale(),
        v2.Resize((32,32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32), # normalizes to range [0,1]
    ])
    letters_array = []
    for i, img in enumerate(char_imgs): # read each character on the plate
        predicted_class, confidence_score, confidence_scores = recognize_character(img, read_model, class_names, transform)
        confidence_scores = dict(sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True))
        print(f'Value: {predicted_class}')
        letters_array.append(predicted_class)
        chars = list(confidence_scores.keys())
        logits = np.array(list(confidence_scores.values()))
        logits = np.array(logits, dtype=np.float64)
        softmax_probs = np.exp(logits) / np.sum(np.exp(logits))
        softmax_confidence_scores = dict(zip(chars, softmax_probs))
        #print(f'Probabilities: {list(softmax_confidence_scores.items())[:4]}')
    return letters_array


@app.route("/detect_read", methods=['GET', 'POST'])
def run_models():
    print("predicting plate image")
    # path parameter
    path =  '../images/data/test/test1.png'
    # detect model parameter
    net = cv2.dnn.readNetFromONNX('../yolov5/runs/train/Model19/weights/best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # read model parameter
    reader = torch.load('../saved_reader_model/net_4_100_/bilinear_images_5x500_16/07-30-2024_18:49:19_0-0106.pth')

    reader = reader.to(device)
    reader.eval()
    # class names parameter
    class_names = dataset.classes
    letters = full_code(path,net,reader,class_names)
    return jsonify({'letters':letters})

if __name__ == '__main__':
    app.run(debug=True)






