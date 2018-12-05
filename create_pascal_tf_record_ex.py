# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script has been modified.
#
# The original can be found here:
#   https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python create_pascal_tf_record_ex.py \
        --annotations_dir=dataset/train/ \
        --label_map_path=dataset/label_map.pbtxt \
        --output_path=train.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('annotations_dir', None,
                    'Path to annotations directory.')
flags.mark_flag_as_required('annotations_dir')
flags.DEFINE_string('images_dir', None, 'Path to images directory. The same as annotations_dir if omitted.')
flags.DEFINE_string('label_map_path', 'label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('output_path', None, 'Path to output TFRecord')
flags.mark_flag_as_required('output_path')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       images_dir,
                       label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    images_dir: Path to image described by the PASCAL XML file
    label_map_dict: A map from string label names to integers ids.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = os.path.join(images_dir, data['filename'])
#  full_path = data['path']
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  annotations_dir = FLAGS.annotations_dir

  images_dir = FLAGS.images_dir
  if not images_dir:
    images_dir = annotations_dir

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  annotations_list = glob.glob(os.path.join(annotations_dir, '*.xml'))
  for idx, xml_file in enumerate(annotations_list):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(annotations_list))
    with tf.gfile.GFile(xml_file, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, images_dir, label_map_dict)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
