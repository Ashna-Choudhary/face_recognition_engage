# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import standard dependencies
import cv2
import os
import tensorflow as tf

#aws
import boto3

#kivy app and layout
class CamApp(App):

    def build(self):
        # Main layout components 
        self.verification_label = Label(size_hint=(1,.1))
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.title_label = Label(text="Face Recognition", size_hint=(1,.2), font_size=22)

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.title_label)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def compare_faces(self, *args):

        sourceFile='D:\FaceID\dapp\dapplication_data\input_image\input_image.jpg'
        targetFile='D:\FaceID\dapp\dapplication_data\input_image\input_image1.jpg'

        client=boto3.client('rekognition')
    
        imageSource=open(sourceFile,'rb')
        imageTarget=open(targetFile,'rb')

        response=client.compare_faces(SimilarityThreshold=80,
                                    SourceImage={'Bytes': imageSource.read()},
                                    TargetImage={'Bytes': imageTarget.read()})
        
        for faceMatch in response['FaceMatches']:
            position = faceMatch['Face']['BoundingBox']
            similarity = str(faceMatch['Similarity'])
            print('The face at ' +
                str(position['Left']) + ' ' +
                str(position['Top']) +
                ' matches with ' + similarity + '% confidence')

        imageSource.close()
        imageTarget.close()     
        return len(response['FaceMatches']) 

    # Verification function to verify person
    def verify(self, *args):

        # save input image from our webcam
        SAVE_PATH=os.path.join('D:', 'FaceID', 'dapp', 'dapplication_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        print("saved image")

        face_matches=self.compare_faces()
        print("Face matches: " + str(face_matches))

        return 0

if __name__ == '__main__':
    CamApp().run()
