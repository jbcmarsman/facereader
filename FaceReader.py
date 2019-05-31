import mysql.connector as mariadb
import os
import pprint
#import cv2

import face_recognition

from pypika import Query, Table, Field

class FaceReader:

    connection = []
    cursor = [] 

    db_host = '192.168.178.2'
    db_user = 'python'
    db_pass = 'raspberry'
    db_port = '3307'
    db_name = 'facereader'

    photo_path = "/nfs/photo/Foto's 2019/Mei/18 mei - Pieterpad route 7 Rolde Tolhek - Schoonloo"
    model_path = "/home/marsman/models/"

    verbosity = False

    def __init__(self):
    
        print "Constructed"
        self.connect()
        print "Connected"            

    def __del__(self):
    
        self.disconnect()
    
    def connect(self):
    
        self.connection = mariadb.connect(user=self.db_user,
                                          password=self.db_pass,
                                          host=self.db_host,
                                          port=self.db_port,
                                          database=self.db_name)   
        self.cursor = self.connection.cursor()
        
    def disconnect(self):

        self.cursor.close()
        self.connection.close()

    def add_to_queue(self, filename):

        photo_queue = Table('photo_queue')
        q =Query.into(photo_queue).columns('filename').insert(filename)
        
        self.cursor.execute(str(q))
        self.connection.commit()
        
    def build_photo_queue(self):
        photos = []

        for (r,d,f) in os.walk(self.photo_path):
            if not '@ea' in r:
                for file in f:
                    if '.jpg' in file.lower():
                        self.add_to_queue(os.path.join(r,file))
    

    def process_queue(self):
        photo_queue = Table('photo_queue')
        q = Query.from_(photo_queue).select(photo_queue.filename).limit(1)
        
        self.cursor.execute(str(q))
        res = self.cursor.fetchone()
        

        self.detect_faces(res[0])

    def get_label_idx(self, label):
        ml = Table("model_labels")
        q = Query.from_(ml).select(ml.idx).where(ml.label == label)

        self.cursor.execute(str(q))
        res = self.cursor.fetchone()

        if not res or res[0] == None:
            print("new label needs assignment.")
            q = Query.into(ml).columns(ml.label).insert(label)
            self.cursor.execute(str(q))
            self.connection.commit()
            res = []
            res.append(self.get_label_idx(label))
        else:
            return res[0]
            
    def detect_faces(self, filename):

        # find lower resolution image
        filebase = os.path.basename(filename)
        filedir  = os.path.dirname(filename)
        minified = filedir + "/@eaDir/" + filebase + "/SYNOPHOTO_THUMB_XL.jpg"
        exist = os.path.isfile(minified)
        print(minified)
        if exist:
            print "mini exist"
            filename = minified
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)
            print(face_locations)
            
    def process_model_photo(self, filename, label):
        pm = Table("photo_models")
        q = Query.from_(pm).select(pm.idx).where(pm.filename == filename)
        self.cursor.execute(str(q))
        res = self.cursor.fetchone()
        
        if not res or res == None:
                image = face_recognition.load_image_file(filename)
                encodings = face_recognition.face_encodings(image)
                label_idx = self.get_label_idx(label)
                print(label + " -> " + str(label_idx))

                for encoding in encodings:
            
                    q = Query.into(pm).columns(pm.label_idx, pm.encoding, pm.filename).insert(label_idx, str(encoding), filename)
                    self.cursor.execute(str(q))
                    self.connection.commit()

    def load_models(self):
        ml = Table("model_labels")
        pm = Table("photo_models")
        
        q = Query.from_(ml).select(ml.idx, ml.label)
        self.cursor.execute(str(q))
        res = self.cursor.fetchall()

        model = {}
        
        for row in res:
            label_idx = row[0]
            label = row[1]
                
            model[label] = []
            nq = Query.from_(pm).select(pm.encoding,pm.weight).where(pm.label_idx == label_idx)
            self.cursor.execute(str(nq))
            nres = self.cursor.fetchall()
            for row in nres:
                model[label].append({ 'encoding' : row[0], 'weight' : row[1]})


        pprint.pprint(model)
                
    def build_model(self):
        for (r,d,f) in os.walk(self.model_path):
            for file in f:
                if '.jpg' in file.lower():
                    label = os.path.basename(r)
                    self.process_model_photo(os.path.join(r,file), label)
        
        
f = FaceReader()
#f.process_queue()
f.load_models()
#f.build_model()
del f
