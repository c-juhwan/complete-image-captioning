# Install pycocotools
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../

# Download COCO dataset
mkdir dataset
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./dataset/
wget http://images.cocodataset.org/zips/train2017.zip -P ./dataset/
wget http://images.cocodataset.org/zips/val2017.zip -P ./dataset/

# Unzip COCO dataset
unzip ./dataset/annotations_trainval2017.zip -d ./dataset/
rm ./dataset/annotations_trainval2017.zip
unzip ./dataset/train2017.zip -d ./dataset/
rm ./dataset/train2017.zip 
unzip ./dataset/val2017.zip -d ./dataset/ 
rm ./dataset/val2017.zip 