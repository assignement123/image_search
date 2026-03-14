## Cai thu vien cho venv
pip install pyefd scikit-image scipy tqdm opencv-python

# image_search

# khoi tao moi truong

docker-compose up -d

# tai thu vien ket noi postgress

pip install psycopg2-binary

## File leaf_debug.py de xem tung anh, extract xem dung chua

python leaf_debug.py 1027.jpg

## File leaf_extract.py de chay ca folder(vi du ten folder : leave), them data, day data anh da extract vao database

python leaf_extract.py --data leaves

## Folder anh cho vao database la chua co cau truc, chi co dang 1 folder co 1 list anh

# Vi du: 1004.py 1005.py 1006.py,....

## Truy van trong database:

SELECT filename
FROM leaf_collection
WHERE filename != '1278.jpg'
ORDER BY efd <-> (
SELECT efd
FROM leaf_collection
WHERE filename = '1278.jpg'
)
LIMIT 5;
