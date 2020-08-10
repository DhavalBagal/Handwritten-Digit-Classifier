import logging, os #logging==0.5.1.2
import torch #torch==1.6.0
import torchvision.transforms as transforms #torchvision==0.7.0
from Model import ConvNet
from flask import Flask, render_template, request, jsonify  # flask==1.1.1

""" Disable all warnings """
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IP_ADDR = "localhost"
PORT = 8000
MODEL_PATH = os.path.dirname(__file__)+'/model.pth'
IMG_SIZE = 28
THRESHOLD = 0.95

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=['POST'])
def classify():
    vec = request.json['img-vector']['img']    #vec = {"img":[[0,10], [2,24], [17,27]]}

    """ Create an IMG_SIZE x IMG_SIZE python list to store all the pixel values of the image """
    img =[[0]*IMG_SIZE for i in range(IMG_SIZE)]
    for coordinates in vec:
        x,y = coordinates
        img[y][x] = 255
    
    """ Convert python list to torch tensor """
    imgVec = torch.FloatTensor(img)
    imgVec = torch.autograd.Variable(imgVec, requires_grad=True)

    """ Add channel dimension to convert (IMG_SIZE, IMG_SIZE) image into (1, IMG_SIZE, IMG_SIZE)  """
    imgVec = imgVec.unsqueeze(0)
    
    ''' Normalizing the image so that all values are between 0-1 and add a batch dimension '''
    imgVec = transform(imgVec).unsqueeze(0)

    """ Feed forward the image tensor to the network and get the predictions """
    outputs = model(imgVec)
    confidence, index = torch.max(outputs.data, dim=1)
    index = index.item()
    confidence = confidence.item()
    
    if confidence<=THRESHOLD:
        res = "Not sure of this one..."
    else:
        res = classes[index]
    
    return jsonify(response=res)

if __name__ == "__main__":

    classes = {0: "Zero (0)", 1:"One (1)", 2:"Two (2)", 3:"Three (3)", 4:"Four (4)", 5:"Five (5)",\
            6:"Six (6)", 7:"Seven (7)", 8:"Eight (8)", 9:"Nine (9)"}

    model = ConvNet(lr=0.001, gpu=False)
    transform = transforms.Normalize([0.5], [0.5])
    device = torch.device('cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
    app.run(host=IP_ADDR, port=PORT, debug=True)