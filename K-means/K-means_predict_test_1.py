import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from display_network import *
from PIL import Image

# Load dữ liệu MNIST
mndata = MNIST(r'D:\Learning-Python\Learning-Basic\MNIST')
mndata.load_testing()
X = mndata.test_images
X0 = np.asarray(X)[:1000, :] / 256.0
X = X0

# Huấn luyện mô hình K-means
K = 10
kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

# Hiển thị các centers của các nhóm
A = display_network(kmeans.cluster_centers_.T, K, 1)
f1 = plt.imshow(A, interpolation='nearest', cmap="jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
fig, axes = plt.subplots(1, 1)

# Chọn ảnh cần nhận diện ký tự
input_image_path = r'D:\Learning-Python\Learning-Basic\test\1.jpg'
input_image = Image.open(input_image_path).convert('L')  # Chuyển đổi ảnh thành ảnh xám
input_image = input_image.resize((28, 28))  # Resize ảnh về kích thước 28x28 pixel
input_data = np.array(input_image) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
input_data = input_data.flatten().reshape(1, -1)  # Biến đổi ảnh thành vector

# # Dự đoán ký tự từ ảnh đầu vào
# pred_cluster = kmeans.predict(input_data)
# nearest_center = kmeans.cluster_centers_[pred_cluster]
# pred_label = pred_label.reshape(-1, 1)
# cluster_indices = np.where(pred_label == pred_cluster)[0]
# character_labels = []
# for idx in cluster_indices:
#     character_labels.append(mndata.test_labels[idx])

# Dự đoán ký tự từ ảnh đầu vào
pred_cluster = kmeans.predict(input_data)
cluster_indices = np.where(pred_label == pred_cluster)[0]
character_labels = []
for idx in cluster_indices:
    character_labels.append(mndata.test_labels[idx])


# Hiển thị kết quả
character_labels = np.array(character_labels)
print("Predicted Cluster:", pred_cluster)
#print("Predicted Character Label:", character_labels)

# Hiển thị ảnh đầu vào
plt.imshow(input_data.reshape(28, 28), cmap='gray')
plt.axis('off')
#plt.show()

N_display = 5  # Số lượng ký tự cần hiển thị trong mỗi nhóm lân cận

# Tìm các ký tự trong lân cận của predicted cluster
predicted_cluster = pred_cluster 
cluster_indices = np.where(pred_label == predicted_cluster)[0]
cluster_samples = X0[cluster_indices]

# Hiển thị các ký tự trong lân cận
plt.figure(figsize=(10, 2))
for i in range(N_display):
    plt.subplot(1, N_display, i + 1)
    plt.imshow(cluster_samples[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
