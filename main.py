from flask import Flask , request , render_template , redirect
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

flag = 0
name = ""
name2 = ""
name3 = ""
percent = 0.0
percent2 = 0.0
percent3 = 0.0
fn = ""


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class_names = ['Ha', 'Kim', 'Lee', 'Ma', 'Song']
class_names2 = ['Sin','IU','Oh','SZ','Park']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classifier')
def classifier():
    global flag
    global name
    global name2
    global name3
    global percent
    global percent2
    global percent3
    global fn
    return render_template('classifier.html', flag = flag, name = name, name2 = name2 , name3 = name3 ,
                                   percent = percent, percent2 = percent2 , percent3 = percent3 ,filename = fn)


@app.route('/fileupload',methods=['POST' , 'GET'])
def fileupload():
    global flag
    global name
    global name2
    global name3
    global percent
    global percent2
    global percent3
    global fn
    if request.method == 'POST':

        f = request.files['file']
        gender = request.form.get('gender')
        print(gender)

        if f.filename == '':
            return redirect(request.url)

        if f and allowed_file(f.filename):

            file = f.filename
            f.save(secure_filename(file))
            base = os.path.splitext(file)[0]
            if not os.path.isfile(base+'.jpg'):
                os.rename(file,base + '.jpg')
            file = (base + '.jpg')
            if os.path.isfile(base+'.png'):
                os.remove(base+'.png')
            if os.path.isfile(base+'.jpeg'):
                os.remove(base+'.jpeg')

            filename = "face_" + file
            filepath = "static/face_" + file

            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
                faces = img[y:y + h, x:x + w]
                cv2.imwrite(filepath, faces)
                cv2.waitKey()

            face = tf.keras.preprocessing.image.load_img(
                filepath, target_size=(180, 180))



            img_array = tf.keras.preprocessing.image.img_to_array(face)
            img_array = tf.expand_dims(img_array, 0)

            if gender == 'M':
                model = tf.keras.models.load_model('model.h5')
                predictions = model.predict(img_array)

                score = tf.nn.softmax(predictions[0])
                sc = np.sort(score)
                scarg = np.argsort(score)

                name = class_names[scarg[4]]
                name2 = class_names[scarg[3]]
                if name2 == 'Kim':
                    name2 = '김수현'
                elif name2 == 'Ma':
                    name2 = '마동석'
                elif name2 == 'Ha':
                    name2 = '하정우'
                elif name2 == 'Lee':
                    name2 = '이정재'
                else:
                    name2 = '송강호'
                name3 = class_names[scarg[2]]
                if name3 == 'Kim':
                    name3 = '김수현'
                elif name3 == 'Ma':
                    name3 = '마동석'
                elif name3 == 'Ha':
                    name3 = '하정우'
                elif name3 == 'Lee':
                    name3 = '이정재'
                else:
                    name3 = '송강호'

                percent = round(100 * sc[4], 2)
                percent2 = round(100 * sc[3], 2)
                percent3 = round(100 * sc[2], 2)

            elif gender == 'F':
                cnn = tf.keras.models.load_model('cnn.h5')
                predictions2 = cnn.predict(img_array)

                score2 = tf.nn.softmax(predictions2[0])
                sc2 = np.sort(score2)
                scarg2 = np.argsort(score2)

                name = class_names2[scarg2[4]]
                name2 = class_names2[scarg2[3]]
                if name2 == 'SZ':
                    name2 = '수지'
                elif name2 == 'IU':
                    name2 = '아이유'
                elif name2 == 'Oh':
                    name2 = '오헌경'
                elif name2 == 'Sin':
                    name2 = '신세경'
                else:
                    name2 = '박은빈'
                name3 = class_names2[scarg2[2]]
                if name3 == 'SZ':
                    name3 = '수지'
                elif name3 == 'IU':
                    name3 = '아이유'
                elif name3 == 'Oh':
                    name3 = '오헌경'
                elif name3 == 'Sin':
                    name3 = '신세경'
                else:
                    name3 = '박은빈'

                percent = round(100 * sc2[4], 2)
                percent2 = round(100 * sc2[3], 2)
                percent3 = round(100 * sc2[2], 2)



            flag = 1
            fn = filename
            print(name,name2,name3,percent,percent2,percent3)
            return render_template('classifier.html', flag=flag, name=name, name2=name2, name3=name3,
                               percent=percent, percent2=percent2, percent3=percent3, filename=fn)
        else:
            return redirect(request.url)
    else:
        flag = 0
        return render_template('classifier.html',flag = flag)




@app.route('/Help')
def Help():
    return render_template('Help.html')


if __name__ == '__main__':
    app.secret_key  = '12345'
    app.run()
