import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class plantseeding:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):
        # load model
        model = load_model('models/model_vgg16.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        y_pred = model.predict(test_image)
        result = np.argmax(y_pred, axis=1)

        if result[0] == 0:
            prediction = 'Black-grass'
            return [{ "image" : prediction}]
        elif result[0] == 1:
            prediction = 'Charlock'
            return [{ "image" : prediction}]
        elif result[0] == 2:
            prediction = 'Cleavers'
            return [{ "image" : prediction}]
        elif result[0] == 3:
            prediction = 'Common Chickweed'
            return [{ "image" : prediction}]
        elif result[0] == 4:
            prediction = 'Common wheat'
            return [{ "image" : prediction}]
        elif result[0] == 5:
            prediction = 'Fat Hen'
            return [{ "image" : prediction}]
        elif result[0] == 6:
            prediction = 'Loose Silky-bent'
            return [{ "image" : prediction}]
        elif result[0] == 7:
            prediction = 'Maize'
            return [{ "image" : prediction}]
        elif result[0] == 8:
            prediction = 'Scentless Mayweed'
            return [{ "image" : prediction}]
        elif result[0] == 9:
            prediction = 'Shepherds Purse'
            return [{ "image" : prediction}]
        elif result[0] == 10:
            prediction = 'Small-flowered Cranesbill'
            return [{ "image" : prediction}]
        else:
            prediction = 'Sugar beet'
            return [{ "image" : prediction}]


