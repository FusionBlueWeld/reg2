import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

root_path = "C:/Users/FUTOSHI/Desktop/ChatGPT_test/06_spatter_prediction_softmax/"

# モデルを読み込む
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# メッシュグリッドを作成する
def create_meshgrid(resolution=100):
    x_range = np.linspace(-6, 6, resolution)
    y_range = np.linspace(30, 600, resolution)
    return np.meshgrid(x_range, y_range)

# データを準備し、正規化する
def prepare_data(mesh_x, mesh_y):
    data = np.vstack((mesh_x.ravel(), mesh_y.ravel())).T
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# 予測を行う
def make_predictions(model, X, resolution):
    # 最も確率の高いクラスを選択
    return model.predict(X).argmax(axis=-1).reshape(resolution, resolution)

# カラーマップを描画する
def plot_heatmap(mesh_x, mesh_y, predictions, original_data):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(mesh_y, mesh_x, predictions, cmap='bwr', shading='auto')
    plt.colorbar(label='Spatter Amount')
    plt.scatter(original_data[:, 1], original_data[:, 0], c=original_data[:, 2], cmap='bwr', edgecolors='k', s=80)
    plt.xlabel('Welding Speed (mm/sec)')
    plt.ylabel('Processing Head Position (mm)')
    plt.xlim(20,610)
    plt.ylim(-6.3,6.3)
    plt.grid()
    plt.savefig(root_path + 'output_data/spatter_heatmap.png')

def load_original_data(file_path):
    data = pd.read_csv(file_path, encoding="shift-jis")
    return data.to_numpy()

if __name__ == '__main__':
    model = load_model(root_path + "output_data/trained_welding_model.h5")
    mesh_x, mesh_y = create_meshgrid()
    X = prepare_data(mesh_x, mesh_y)
    predictions = make_predictions(model, X, 100)
    original_data = load_original_data(root_path + "input_data/input_data.csv")
    plot_heatmap(mesh_x, mesh_y, predictions, original_data)