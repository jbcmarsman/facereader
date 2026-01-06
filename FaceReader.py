import time
from datetime import datetime

now= time.time()
import logging

# setup logging
logging.basicConfig(filename='reader.log', filemode='w',format='[%(asctime)s][%(levelname)s] %(message)s',level=logging.INFO )
logging.info("[lib] logging took " + str(time.time()-now) + " seconds")

now= time.time()
import mysql.connector as mariadb
logging.info("[lib] mysql took " + str(time.time()-now) + " seconds")

now= time.time()
import os
logging.info("[lib] os took " + str(time.time()-now) + " seconds")

now= time.time()
import numpy as np
logging.info("[lib] numpy took " + str(time.time()-now) + " seconds")

now= time.time()
import pprint
logging.info("[lib] pprint took " + str(time.time()-now) + " seconds")

now= time.time()
now= time.time()
import cv2
logging.info("[lib] openCV took " + str(time.time()-now) + " seconds")

now= time.time()
import face_recognition
logging.info("[lib] face_recognition took " + str(time.time()-now) + " seconds")

now= time.time()
from pypika import Query, Table, Field
from pypika import functions as fn
logging.info("[lib] pypika took " + str(time.time()-now) + " seconds")

class PhotoPathBuilder:
    root = "/nfs/photo/Foto's 2020" 
    maxdepth = 2
    queue = []
            
    def __init__(self, maxdepth=2):
        self.maxdepth = maxdepth
        depth = 0
        for [root, dirnames, filenames] in os.walk(self.root):
            if depth <= self.maxdepth:

                if dirnames:
                    dirnames.remove('@eaDir')
                    for d in dirnames:
                         self.queue.append(os.path.join(root,d))
                         try:
                             self.queue.remove(root)
                         except ValueError:
                             pass
            depth=depth+1                     
        logging.info('[pathbuilder] ' + str(len(self.queue)) + ' directories found')
        
    def get(self):
        return self.queue

    def print_queue(self):
        pprint.pprint(self.queue)


class FaceReader:

    connection = []
    cursor = [] 

    db_host = '192.168.178.2'
    db_user = 'python'
    db_pass = '******'
    db_port = '3307'
    db_name = 'facereader'

    photo_path = "" #Mei/18 mei - Pieterpad route 7 Rolde Tolhek - Schoonloo"
    model_path = "/home/marsman/models/"

    face_detection_threshold = 0.55
    verbosity = 3

    model = {}

    # constructor
    def __init__(self, path='/nfs/photo/Foto\'s 2019'):
        logging.info('[facereader] initialized with path ' + path)
        self.photo_path = path
        self.connect()
        self.load_models()

    # destructor
    def __del__(self):
    
        self.disconnect()

    # connect to MariaDB
    def connect(self):
    
        self.connection = mariadb.connect(user=self.db_user,
                                          password=self.db_pass,
                                          host=self.db_host,
                                          port=self.db_port,
                                          database=self.db_name)   
        self.cursor = self.connection.cursor()

    # disconnect from MariaDB
    def disconnect(self):

        self.cursor.close()
        self.connection.close()

    # add row to photo_queue table
    def add_to_queue(self, filename):

        photo_queue = Table('photo_queue')
        q =Query.into(photo_queue).columns('filename').insert(filename)
        
        self.cursor.execute(str(q))
        self.connection.commit()

    # traverse photo_path to find jpg files, and add them to queue
    def build_photo_queue(self):
        photos = []

        for (r,d,f) in os.walk(self.photo_path):
            if not '@ea' in r:
                for file in f:
                    if '.jpg' in file.lower():
                        self.add_to_queue(os.path.join(r,file))
            

    # after processing, remove from queue
    def remove_photo_from_queue(self, filename):
        pq = Table("photo_queue")
        q = Query.from_(pq).delete().where(pq.filename == filename)

        self.cursor.execute(str(q))
        self.connection.commit()
        
    
    # search for label, given an index
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

    # after a match has been found, store label, photo and score 
    def insert_photo(self, filename):
        ip = Table("indexed_photos")
        q = Query.into(ip).columns('filename').insert(filename);
        self.cursor.execute(str(q))
        self.connection.commit()

        q = Query.from_(ip).select(fn.Max(ip.idx))
        self.cursor.execute(str(q))
        res = self.cursor.fetchone()
        return res[0]
        
    # after a match has been found, store label, photo and score 
    def insert_photo_match(self, photo_idx, label_idx, score):
        lm = Table("label_matches")
        q = Query.into(lm).columns('photo_idx', 'label_idx', 'score').insert(photo_idx, label_idx, score)
        self.cursor.execute(str(q))
        self.connection.commit()

    # process photo's in the model folder (targets for recognition)
    def process_model_photo(self, filename, label):
        pm = Table("photo_models")
        q = Query.from_(pm).select(pm.idx).where(pm.filename == filename)
        self.cursor.execute(str(q))
        res = self.cursor.fetchone()
        
        if not res or res == None:
                image = face_recognition.load_image_file(filename)
                encodings = face_recognition.face_encodings(image)
                label_idx = self.get_label_idx(label)
                logging.debug("[facereader] " + label + " -> " + str(label_idx))

                for encoding in encodings:
            
                    q = Query.into(pm).columns(pm.label_idx, pm.encoding, pm.filename).insert(label_idx, str(encoding), filename)
                    self.cursor.execute(str(q))
                    self.connection.commit()

    # load model data
    def load_models(self):

        ml = Table("model_labels")
        pm = Table("photo_models")
        
        q = Query.from_(ml).select(ml.idx, ml.label)
        self.cursor.execute(str(q))
        res = self.cursor.fetchall()

        item_count = 0 
        for row in res:
            now = time.time()
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

            if self.verbosity > 1:
                logging.info("[loading model data] " + label + " took " + str(time.time()-now) + " seconds")
                
    # (re)build model data from model directory
    def build_model(self):
        for (r,d,f) in os.walk(self.model_path):
            for file in f:
                if '.jpg' in file.lower():
                    label = os.path.basename(r)
                    self.process_model_photo(os.path.join(r,file), label)
        
    # process photo's in the queue to match for faces
    def process_queue(self):
        pq = Table("photo_queue")
        ip = Table("indexed_photos")
        
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

            image = face_recognition.load_image_file(str(minified))
            current_encoding = face_recognition.face_encodings(image)

            ins_idx = self.insert_photo(filename)

            m=0
            if current_encoding:                
                for sub_encoding in current_encoding:
                    result = face_recognition.face_distance(enc_mat_np, sub_encoding)

                    i=0
                    for sub_result in result:
                
                        if sub_result < self.face_detection_threshold:
                            label_idx = self.get_label_idx(enc_labels[i])  
                            self.insert_photo_match(ins_idx, label_idx, sub_result)
                            m=m+1
                        i=i+1
                
            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
            logging.debug("[ " + timestamp + " ] [ " + str(m) + " matches ] " + filename)
            
            self.remove_photo_from_queue(filename)

pb = PhotoPathBuilder()

for dirname in pb.get():

    f = FaceReader(dirname)
    f.build_photo_queue()
    f.process_queue()
    del f

            

