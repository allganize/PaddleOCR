FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
WORKDIR /paddle_ocr/

RUN apt-get update -y
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN python3.8 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

RUN apt-get -y install python3.8-dev
RUN apt-get -y install libgl1-mesa-glx
RUN pip install paddlepaddle-gpu==2.3.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

RUN pip install shapely==1.8.5.post1
RUN pip install scikit-image==0.19.3
RUN pip install imgaug==0.4.0
RUN pip install pyclipper==1.3.0.post4
RUN pip install lmdb==1.3.0
RUN pip install tqdm==4.64.1
RUN pip install numpy==1.23.5
RUN pip install visualdl==2.4.1
RUN pip install rapidfuzz
RUN pip install opencv-python==4.6.0.66
RUN pip install opencv-contrib-python==4.6.0.66
RUN pip install cython==0.29.32
RUN pip install lxml==4.9.1
RUN pip install premailer==3.10.0
RUN pip install openpyxl==3.0.10
RUN pip install attrdict==2.0.1
RUN pip install Polygon3==3.0.9.1
RUN pip install lanms-neo==1.0.2
RUN pip install PyMuPDF==1.21.0
RUN pip install protobuf==3.20.3

RUN pip install torch==1.13.0
RUN pip install torchvision==0.14.0
RUN pip install transformers==4.24.0
RUN pip install timm==0.6.11

COPY . .

# CMD ["bash", "./script/cmd.sh"]
CMD tail -f /dev/null