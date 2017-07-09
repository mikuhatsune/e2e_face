#mkdir model
#mkdir log
export GLOG_log_dir=.
~/caffe-master/build/tools/caffe train --solver=solver.prototxt --gpu=0