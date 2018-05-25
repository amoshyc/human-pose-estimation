download-mpii:
    mkdir mpii
    wget 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz' -O ./mpii/mpii.tar.gz
	wget 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip' -O ./mpii/anno.zip
	python -c "from mpii import MPIIParse; MPIIParse('./mpii')"

download-coco:
	mkdir -p coco/images/train2017
	gsutil -m rsync gs://images.cocodataset.org/train2017 coco/images/train2017
	mkdir -p coco/images/val2017
	gsutil -m rsync gs://images.cocodataset.org/val2017 coco/images/val2017
	mkdir -p coco/annotations
	gsutil -m rsync gs://images.cocodataset.org/annotations coco/annotations
