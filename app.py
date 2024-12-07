from PIL import Image
from datetime import date, timedelta
from flask_cors import CORS, cross_origin
from flask import Flask, request, send_file, render_template, jsonify

import scripts.down as down
from scripts import predict
from scripts.utils import to_our, reset_path, pad_image, normalize, ALLOWED_EXTENSIONS, TMP_PATH

app = Flask(__name__)

CORS(app, support_credentials=True)

model = predict.make_model()

@app.route('/')
def index():
    return render_template('index.html')

# when user sends image
@app.route("/post-image", methods=['POST'])
@cross_origin(supports_credentials=True)
def handle_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image in request'}), 400
    
    file = request.files['image']
    ext = file.filename.split('.')[-1]

    if file and ext in ALLOWED_EXTENSIONS:
        
        #delete the tmp folder and remake it
        reset_path()

        f_name = f'{TMP_PATH}/tmp.{ext}'

        file.save(f_name)

        img = predict.preprocess(f_name)
        msk = predict.predict(model, img)

        img = predict.mix_all(f_name, msk)

        img.save(f'{TMP_PATH}/tmp.png')

        return send_file(f'{TMP_PATH}/tmp.png', mimetype='image/png', as_attachment=True)
    
    return "File type not allowed", 400


# when user uses map
@app.route("/post-map", methods=['POST'])
@cross_origin(supports_credentials=True)
def handle_map():
    if request.method == 'POST':
       
        poly = to_our(request.json)

        end_date = date(2024,1,21)
        start_date = end_date - timedelta(20)
        
        reset_path()

        dat = down.get_catalogue(start_date, end_date, aoi=poly)

        s3_uri, pref= down.get_s3_uri(dat)

        img = down.get_s3_content(s3_uri, pref)

        f_name = down.get_roi(img, poly)

        img = predict.preprocess(f_name)
        
        p_img = pad_image(img)
        
        msk = predict.predict(model, p_img)

        p_img = normalize(p_img.cpu().numpy(), 0, 255).transpose((1,2,0)).astype('uint8')        

        img = predict.mix_all(Image.fromarray(p_img).convert('RGBA'), msk)

        img.save(f'{TMP_PATH}/tmp.png')

        return send_file(f'{TMP_PATH}/tmp.png', mimetype='image/png', as_attachment=True)
    
    return "File type not allowed", 400

