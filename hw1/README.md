# pdm-f24-hw1

NYCU Perception and Decision Making 2024 Fall

Spec: [Google Docs](https://docs.google.com/document/d/1QSbSWJ7s78h9QRS4EC3gsECFF8JDg0IT/edit?usp=sharing&ouid=101044242612677438105&rtpof=true&sd=true)

## Preparation
The replica dataset, you can use the same one in `hw0`.

## Collect data
```
python load.py -f 1 # first floor
python load.py -f 2 # second floor
```
## Run Task1
```
python bev.py
```
## Run Task2
```
python reconstruct.py -v open3d -f 1 # use open3d on first floor
python reconstruct.py -v open3d -f 2 # use open3d on second floor
python reconstruct.py -v my_icp -f 1 # use my_icp on first floor
python reconstruct.py -v my_icp -f 2 # use my_icp on second floor
```