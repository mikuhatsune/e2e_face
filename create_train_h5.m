%detect_faces('data/webface_list.txt', '../CASIA-WebFace', 'webface', 'data/webface_raw_bbox.txt');
%detect_faces('data/lfw_list.txt', '../lfw', 'lfw', 'data/lfw_raw_bbox.txt');

crop_and_create_h5('data/webface_raw_bbox.txt', '../CASIA-WebFace', 'data', 'webface', 1.3);