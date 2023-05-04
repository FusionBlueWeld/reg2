import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# GPU を使用しない設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ニューラルネットのパラメータ初期値を固定
tf.random.set_seed(42)

# ルートパスの設定
root_path = "C:/Users/FUTOSHI/Desktop/ChatGPT_test/06_spatter_prediction_softmax/"


def preprocess_data(data_path):
    # データの読み込み（Shift-JISでエンコードされたCSVファイル）
    data = pd.read_csv(data_path, encoding="shift-jis")
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, 2].values

    # データの標準化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # カテゴリカル変数への変換
    y = to_categorical(y, num_classes=3)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    # モデルの構築
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # モデルのコンパイル
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_val, y_val):
    # モデルの学習
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0)

def save_model(model, model_path):
    # モデルの保存
    model.save(model_path)

def plot_learning_curve(history):
    # 学習曲線のプロット
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # MSEプロット
    ax1.plot(history.history['loss'], color='red', label='MSE (Train)')
    ax1.plot(history.history['val_loss'], color='red', linestyle='dashed', label='MSE (Validation)')

    # R^2プロット
    ax2.plot(history.history['accuracy'], color='blue', label='R^2 (Train)')
    ax2.plot(history.history['val_accuracy'], color='blue', linestyle='dashed', label='R^2 (Validation)')

    # 凡例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.7, 1))

    # グラフの設定
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax2.set_ylabel('R^2')
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim(0, 1)
    ax1.grid()
    fig.tight_layout()
    plt.savefig(root_path + 'output_data/learning_curve.png')



if __name__ == '__main__':
    X_train, X_val, y_train, y_val = preprocess_data(root_path + "input_data/input_data.csv")
    model = create_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    save_model(model, root_path + "output_data/trained_welding_model.h5")
    plot_learning_curve(history)

    # 最後のエポックのMSEとR^2を表示
    final_train_mse = history.history['loss'][-1]
    final_val_mse = history.history['val_loss'][-1]
    final_train_r2 = history.history['accuracy'][-1]
    final_val_r2 = history.history['val_accuracy'][-1]

    print("Train MSE: {:.4f}".format(final_train_mse))
    print("Validation MSE: {:.4f}".format(final_val_mse))
    print("Train R^2: {:.4f}".format(final_train_r2))
    print("Validation R^2: {:.4f}".format(final_val_r2))
