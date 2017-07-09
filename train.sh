mkdir model
mkdir log
export GLOG_log_dir=log
~/caffe-master/build/tools/caffe train --solver=solver.prototxt --gpu=0 --weights shallow/shallow_iter_30000.caffemodel