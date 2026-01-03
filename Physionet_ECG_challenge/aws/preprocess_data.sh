
sudo yum install -y zip unzip python3 python3-pip
pip install kaggle awscli

# For large datasets, using memory optimized with instance store volume+redhat os led to a fast download
# Create Iam Role to upload to S3
############################################################
sudo mkdir -p $HOME/data
sudo  cd $HOME/data
sudo touch kaggle_download.log
# 1) Create filesystem
sudo mkfs -t ext4 -F /dev/nvme1n1

# 2) Mount instance store volume
mkdir -p /home/ec2-user/data
sudo mount -t ext4 /dev/nvme1n1 /home/ec2-user/data
chown ec2-user:ec2-user /home/ec2-user/data
sudo chmod 777 /home/ec2-user/data
df -h

# Download data from Kaggle
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c physionet-ecg-image-digitization


aws s3 cp ./physionet-ecg-image-digitization.zip  \
  s3://kaggle-challenge-ayoubmk/physionet/physionet.zip

# Generate a 1h pre-signed URL (run where AWS creds exist)
aws s3 presign s3://kaggle-challenge-ayoubmk/physionet/physionet.zip --expires-in 7200

# Download using the URL (no AWS creds needed)
wget -O /content/physionet.tar.zst.zip "<PASTE_PRESIGNED_URL_HERE>"


##############################################################
##############################################################
##############################################################
# Extract small files and folders outside excluding (the big) train folder
ZIP=/data/physionet-ecg-image-digitization.zip
OUT=/data/filtered

mkdir -p "$OUT"

# Extract everything except train/
unzip -q "$ZIP" -d "$OUT" -x "train/*"
##############################################################
# Get image folder names
unzip -l "$ZIP" \
  | awk '{print $4}' \
  | grep '^train/' \
  | cut -d/ -f2 \
  | grep -v '^$' \
  | sort -u > /data/train_folders.txt

head /data/train_folders.txt
wc -l /data/train_folders.txt
##############################################################


# Extract the first image and csv file from each ECG folder
ZIP=physionet-ecg-image-digitization.zip
OUT=/data/filtered
TMP=/data/tmp
cd /data/
mkdir -p "$TMP"
mkdir -p "$OUT/train"

while read -r ECG; do
  echo "Processing $ECG"

  # 1) Extract only this folder into TMP
  unzip -q "$ZIP" "train/$ECG/*" -d "$TMP"

  # 2) Create destination
  mkdir -p "$OUT/train/$ECG"

  # 3) Keep only CSV + first image (0001)
  cp "$TMP/train/$ECG/"*.csv "$OUT/train/$ECG/" 2>/dev/null || true
  cp "$TMP/train/$ECG/"*0001*.png "$OUT/train/$ECG/" 2>/dev/null || true

  # 4) Delete temp immediately (keeps disk usage bounded)
  rm -rf "$TMP/train/$ECG"

done < train_folders.txt
##############################################################
# Check nothing was forgotten
if [ "$expected" -eq "$actual" ]; then
  echo "✅ OK: folder counts match ($actual)"
else
  echo "❌ MISMATCH: expected $expected, got $actual"
fi
##############################################################

# Send data to S3
cd /data
zip -r physionet_filtered.zip filtered
aws s3 cp /data/physionet_filtered.zip \
  s3://kaggle-challenge-ayoubmk/physionet/physionet_filtered.zip
