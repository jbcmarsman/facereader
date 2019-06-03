import mysql.connector as mariadb
import os
import numpy as np
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

    face_detection_threshold = 0.55
    verbosity = False

    model = {}
    
    def __init__(self):
        
        self.connect()
        self.load_models()
        
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
    


    def remove_photo_from_queue(self, filename):
        pq = Table("photo_queue")
        q = Query.from_(pq).delete().where(pq.filename == filename)

        self.cursor.execute(str(q))
        self.connection.commit()
        
        
    def process_queue(self):
        pq = Table('photo_queue')
        q = Query.from_(pq).select(pq.filename).where(pq.filename.like('%DSC_0087%'))
        print(str(q))
        
        self.cursor.execute(str(q))
        res = self.cursor.fetchall()        

        for row in res:
            filename = row[0]
            self.detect_faces(filename)

    def get_label_idx(self, label):
        ml = Table("model_labels")
        q = Query.from_(ml).select(ml.idx).where(ml.label == label)

        self.cursor.execute(str(q))
        res = self.cursor.fetchone()

        if not res or res[0] == None:
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

    def insert_photo_match(self, photo_idx, label_idx, score):
        lm = Table("label_matches")
        q = Query.into(lm).columns('photo_idx', 'label_idx', 'score').insert(photo_idx, label_idx, score)
        print(str(q))
        self.cursor.execute(str(q))
        self.connection.commit()

        
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
        print("Loading models...")
        ml = Table("model_labels")
        pm = Table("photo_models")
        
        q = Query.from_(ml).select(ml.idx, ml.label)
        self.cursor.execute(str(q))
        res = self.cursor.fetchall()
        
        for row in res:
            label_idx = row[0]
            label = row[1]
                
            self.model[label] = []
            nq = Query.from_(pm).select(pm.encoding,pm.weight).where(pm.label_idx == label_idx)
            self.cursor.execute(str(nq))
            nres = self.cursor.fetchall()
            for row in nres:
                
                enc_str = np.array(str(row[0]).replace('[','').replace(']','').split())
                encoding = enc_str.astype(np.float)
                self.model[label].append({ 'encoding' : encoding, 'weight' : row[1]})
                        
    def build_model(self):
        for (r,d,f) in os.walk(self.model_path):
            for file in f:
                if '.jpg' in file.lower():
                    label = os.path.basename(r)
                    self.process_model_photo(os.path.join(r,file), label)
        

    def process_queue(self):
        pq = Table("photo_queue")
        q = Query.from_(pq).select(pq.idx, pq.filename)
        
        self.cursor.execute(str(q))
        res = self.cursor.fetchall()

        enc_mat = []
        enc_labels = []

        for item in self.model.keys():
            for d in self.model[item]:
                encoding = d['encoding']
                enc_mat.append(encoding)
                enc_labels.append(item)
                
        enc_mat_np = np.asarray(enc_mat)

        for row in res:
            photo_idx= row[0]
            filename = row[1]
            filebase = os.path.basename(filename)
            filedir  = os.path.dirname(filename)        

            minified = filedir + "/@eaDir/" + filebase + "/SYNOPHOTO_THUMB_XL.jpg"
        
            print("Processing: " + filename)
            image = face_recognition.load_image_file(str(minified))
            current_encoding = face_recognition.face_encodings(image)
            
            result = face_recognition.face_distance(enc_mat_np, current_encoding)

            i=0
            for sub_result in result:
                q = Query.insert()
                if sub_result < self.face_detection_threshold:
                    label_idx = self.get_label_idx(enc_labels[i])  
                    self.insert_photo_match(photo_idx, label_idx, sub_result)
                i=i+1

            self.remove_photo_from_queue(filename)
        
f = FaceReader()
#f.process_queue()

#f.build_model()
f.process_queue()
del f
