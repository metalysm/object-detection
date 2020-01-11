
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops


import imageio  #Video içerisinde işlem yapmak için. Yüz tanımadaki gibi.

reader = imageio.get_reader('video/1.mp4')  #Video klasörüne 1 adlı videoya bakıcak ve fps alıp üzerinde işlem yapıp output olarak çıkaracak.
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('video/output.mp4', fps=fps)


from utils import label_map_util

from utils import visualization_utils as vis_util


# Hangi model olduğu.
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Model yükleniyor.


#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
 # file_name = os.path.basename(file.name)
  #if 'frozen_inference_graph.pb' in file_name:
   # tar_file.extract(file, os.getcwd())


# ## Tensorflow model hafızaya atılıyor.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Label map yükleniyor.
#Label, map indekslerini kategori adlarıyla eşleştirir, böylece erişim ağımız indirme işlemini tahmin ettiğinde bunun 'uçağa' karşılık geldiğini biliyoruz. Burada dahili yardımcı fonksiyonları kullanıyoruz, ancak uygun dize etiketlerine bir sözlük eşleştirme tamsayısı döndüren herhangi bir şey iyi olacaktır.



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Yardımcı Kod



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Tanıma Kodu


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():

      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
          # Bu işlem sadece bir fotoğraf için.
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Maskeyi kutu koordinatlarından resim koordinatlarına dönüştürmek ve resim boyutuna sığdırmak için reframe gerekiyor.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Düzeni takip edip yığıt boyutunu geri ekliyoruz.
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Sonucu çalıştır.
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # bütün çıktılar numpy32 dizileridir. Bu yüzden type'ları uygun şekilde dönüştürüyoruz.
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

with tf.Session(graph=detection_graph) as sess:  #Her döngüde yeniden session açılması işlemleri baya yavaşlatacağı için
    for i, image_np in enumerate(reader): #döngüyü session içine alıyoruz böylece session sürekli açılmamış olacak.
      #Video içerisinde her frame için döngü oluşturuyoruz.
      image_np_expanded = np.expand_dims(image_np, axis=0)
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      writer.append_data(image_np) #image_np üzerinde işlemi yapıp tekrar image_np'ye atıyor.
      print(i)  #Kaçıncı frame üzerinde işlem yapılıyor onu görüyoruz.
    writer.close()

