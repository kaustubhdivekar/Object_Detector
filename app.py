from flask import Flask, request, send_file, render_template
from ultralyticsplus import YOLO, render_result
import requests
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

@app.route("/")

def home():
    return render_template("home.html")

@app.route('/test', methods=['GET', 'POST'])

def test():
    # load model
    model = YOLO('ultralyticsplus/yolov8s')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # set image
    # req_Json = request.json
    # image = req_Json['image']
    # https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg
    # https://cdn.roadtrips.com/wp-content/uploads/2017/09/semi-final-1920x1063.jpg

    if request.method == 'POST':
        image = request.form.get('image')

    # perform inference
    results = model.predict(image)

    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image, result=results[0])
    # render.show()

    # render to body via send_file with reading from file on disk
    # tempPath = "images.jpg"
    # render.save(tempPath)
    # return send_file(tempPath,mimetype='image/jpeg',as_attachment=True)

    # Save the rendered image to a bytes buffer
    buf = io.BytesIO()
    render.save(buf, format='JPEG')
    buf.seek(0)
    # Return the image as a file attachment
    return send_file(buf, mimetype='image/jpeg')


    # if request.method == "GET":
    #     return jsonify({"response": "Get Request Called"})
    # elif request.method == "POST":
    #     req_Json = request.json
    #     name = req_Json['name']
    #     return jsonify({"response": "Hi " + name})

@app.route('/test2', methods=['GET','POST'])

def test2():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # input from user
    # req_Json = request.json
    # image = req_Json['image']

    if request.method == 'POST':
        # set image
        image = request.form.get('image')


    # image = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(image, stream=True).raw).convert('RGB')

    # conditional image captioning
    # text = "a photography of"
    # inputs = processor(raw_image, text, return_tensors="pt")

    # out = model.generate(**inputs)
    # print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    # render to body
    return (processor.decode(out[0]))


if __name__ == '__main__':
    app.run(debug=True, port=9090)