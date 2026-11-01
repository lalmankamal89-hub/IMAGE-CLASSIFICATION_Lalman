{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/lalmankamal89-hub/IMAGE-CLASSIFICATION_Lalman/blob/main/4_img_classification_Lalman.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "d8775389",
      "metadata": {
        "id": "d8775389"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Load the data\n",
        "from keras.datasets import cifar10\n",
        "(x_train, y_train), (x_test, y_test) =cifar10.load_data()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "efhgntHwpVwY",
        "outputId": "6ab9a896-6b19-458f-ccf6-fa8775ec78e4"
      },
      "id": "efhgntHwpVwY",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n",
            "\u001b[1m170498071/170498071\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1275s\u001b[0m 7us/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "0007ccac",
      "metadata": {
        "id": "0007ccac"
      },
      "outputs": [],
      "source": [
        "# CNN model\n",
        "model = tf.keras.Sequential([\n",
        "    tf.keras.layers.Rescaling(1./255),\n",
        "    tf.keras.layers.Conv2D(16, (3,3), activation=\"relu\"),\n",
        "    tf.keras.layers.MaxPooling2D(),\n",
        "    tf.keras.layers.Flatten(),\n",
        "    tf.keras.layers.Dense(10, activation=\"softmax\")\n",
        "])\n",
        "# Compile\n",
        "model.compile(\n",
        "    optimizer=\"adam\",\n",
        "    loss=\"sparse_categorical_crossentropy\",\n",
        "    metrics=[\"accuracy\"]\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "bc8d5f7c",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "bc8d5f7c",
        "outputId": "d95ce8e3-cfbc-4af6-b05b-88f965f0acb0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.4503 - loss: 1.5579\n",
            "Epoch 2/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 11ms/step - accuracy: 0.5564 - loss: 1.2743\n",
            "Epoch 3/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.5891 - loss: 1.1799\n",
            "Epoch 4/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.6111 - loss: 1.1207\n",
            "Epoch 5/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.6295 - loss: 1.0725\n",
            "Epoch 6/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.6441 - loss: 1.0367\n",
            "Epoch 7/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.6543 - loss: 1.0051\n",
            "Epoch 8/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 11ms/step - accuracy: 0.6636 - loss: 0.9784\n",
            "Epoch 9/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.6689 - loss: 0.9578\n",
            "Epoch 10/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.6766 - loss: 0.9385\n",
            "Epoch 11/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 11ms/step - accuracy: 0.6779 - loss: 0.9265\n",
            "Epoch 12/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.6875 - loss: 0.9090\n",
            "Epoch 13/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.6890 - loss: 0.8965\n",
            "Epoch 14/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.6941 - loss: 0.8837\n",
            "Epoch 15/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 10ms/step - accuracy: 0.6981 - loss: 0.8722\n",
            "Epoch 16/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20s\u001b[0m 10ms/step - accuracy: 0.7000 - loss: 0.8638\n",
            "Epoch 17/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 11ms/step - accuracy: 0.7031 - loss: 0.8534\n",
            "Epoch 18/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7080 - loss: 0.8465\n",
            "Epoch 19/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 10ms/step - accuracy: 0.7098 - loss: 0.8356\n",
            "Epoch 20/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 10ms/step - accuracy: 0.7111 - loss: 0.8304\n",
            "Epoch 21/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7160 - loss: 0.8221\n",
            "Epoch 22/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7176 - loss: 0.8142\n",
            "Epoch 23/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20s\u001b[0m 10ms/step - accuracy: 0.7190 - loss: 0.8077\n",
            "Epoch 24/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7205 - loss: 0.8017\n",
            "Epoch 25/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7203 - loss: 0.8007\n",
            "Epoch 26/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7249 - loss: 0.7907\n",
            "Epoch 27/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7274 - loss: 0.7850\n",
            "Epoch 28/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7286 - loss: 0.7803\n",
            "Epoch 29/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7289 - loss: 0.7767\n",
            "Epoch 30/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7317 - loss: 0.7705\n",
            "Epoch 31/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20s\u001b[0m 10ms/step - accuracy: 0.7328 - loss: 0.7672\n",
            "Epoch 32/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 9ms/step - accuracy: 0.7322 - loss: 0.7661\n",
            "Epoch 33/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7348 - loss: 0.7570\n",
            "Epoch 34/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 9ms/step - accuracy: 0.7396 - loss: 0.7511\n",
            "Epoch 35/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 10ms/step - accuracy: 0.7396 - loss: 0.7483\n",
            "Epoch 36/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7386 - loss: 0.7477\n",
            "Epoch 37/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7396 - loss: 0.7426\n",
            "Epoch 38/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7410 - loss: 0.7390\n",
            "Epoch 39/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 9ms/step - accuracy: 0.7436 - loss: 0.7373\n",
            "Epoch 40/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20s\u001b[0m 9ms/step - accuracy: 0.7433 - loss: 0.7359\n",
            "Epoch 41/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7444 - loss: 0.7303\n",
            "Epoch 42/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 10ms/step - accuracy: 0.7442 - loss: 0.7252\n",
            "Epoch 43/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 9ms/step - accuracy: 0.7485 - loss: 0.7212\n",
            "Epoch 44/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20s\u001b[0m 9ms/step - accuracy: 0.7483 - loss: 0.7181\n",
            "Epoch 45/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7472 - loss: 0.7180\n",
            "Epoch 46/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7482 - loss: 0.7154\n",
            "Epoch 47/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 10ms/step - accuracy: 0.7533 - loss: 0.7099\n",
            "Epoch 48/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7514 - loss: 0.7087\n",
            "Epoch 49/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 10ms/step - accuracy: 0.7527 - loss: 0.7042\n",
            "Epoch 50/50\n",
            "\u001b[1m1563/1563\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 10ms/step - accuracy: 0.7539 - loss: 0.7024\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1mModel: \"sequential_2\"\u001b[0m\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential_2\"</span>\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ rescaling_2 (\u001b[38;5;33mRescaling\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m3\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m16\u001b[0m)     │           \u001b[38;5;34m448\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m16\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten_2 (\u001b[38;5;33mFlatten\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3600\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_2 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m)             │        \u001b[38;5;34m36,010\u001b[0m │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ rescaling_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Rescaling</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">448</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3600</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>)             │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,010</span> │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m109,376\u001b[0m (427.25 KB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">109,376</span> (427.25 KB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m36,458\u001b[0m (142.41 KB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">36,458</span> (142.41 KB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Optimizer params: \u001b[0m\u001b[38;5;34m72,918\u001b[0m (284.84 KB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Optimizer params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">72,918</span> (284.84 KB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        }
      ],
      "source": [
        "# Redefine model with 10 outputs for CIFAR-10 classes\n",
        "model = tf.keras.Sequential([\n",
        "    tf.keras.layers.Rescaling(1./255),\n",
        "    tf.keras.layers.Conv2D(16, (3,3), activation=\"relu\"),\n",
        "    tf.keras.layers.MaxPooling2D(),\n",
        "    tf.keras.layers.Flatten(),\n",
        "    tf.keras.layers.Dense(10, activation=\"softmax\")\n",
        "])\n",
        "\n",
        "# Compile\n",
        "model.compile(\n",
        "    optimizer=\"adam\",\n",
        "    loss=\"sparse_categorical_crossentropy\",\n",
        "    metrics=[\"accuracy\"]\n",
        ")\n",
        "\n",
        "# Train\n",
        "model.fit(x_train, y_train, epochs=50)\n",
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "c1e73612",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c1e73612",
        "outputId": "dfe5f5c0-8612-4869-95cb-8d8d03f374a1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 38ms/step\n",
            "Predicted class: cat\n",
            "Actual class: cat\n"
          ]
        }
      ],
      "source": [
        "# Prediction using a sample from x_test\n",
        "import numpy as np\n",
        "\n",
        "# Define the CIFAR-10 class names\n",
        "cifar10_classes = [\n",
        "    'airplane', 'automobile', 'bird', 'cat', 'deer',\n",
        "    'dog', 'frog', 'horse', 'ship', 'truck'\n",
        "]\n",
        "\n",
        "# Get a test sample image and preprocess it\n",
        "# CIFAR-10 images are already 32x32, so we expand the dimensions to match batch shape (1, 32, 32, 3)\n",
        "sample_idx = 0\n",
        "x = np.expand_dims(x_test[sample_idx], axis=0)\n",
        "\n",
        "# Predict\n",
        "p = model.predict(x)\n",
        "predicted_class = cifar10_classes[p.argmax()]\n",
        "actual_class = cifar10_classes[y_test[sample_idx][0]]\n",
        "\n",
        "print(f\"Predicted class: {predicted_class}\")\n",
        "print(f\"Actual class: {actual_class}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "e56dfef8",
      "metadata": {
        "id": "e56dfef8"
      },
      "outputs": [],
      "source": [
        "model.save('CAT.keras')\n",
        "\n",
        "import json\n",
        "\n",
        "# Define the actual CIFAR-10 class names\n",
        "cifar10_classes = [\n",
        "    'airplane', 'automobile', 'bird', 'cat', 'deer',\n",
        "    'dog', 'frog', 'horse', 'ship', 'truck'\n",
        "]\n",
        "\n",
        "# Save class names to a JSON file\n",
        "with open(\"classes.json\", \"w\") as f:\n",
        "    json.dump(cifar10_classes, f)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "609f0f3a",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "609f0f3a",
        "outputId": "5943012f-4ea6-4228-cf24-6046d17bbb38"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 55ms/step\n",
            "[[1.5322734e-03 1.4658409e-04 1.4667632e-03 6.9535106e-01 2.8653117e-04\n",
            "  2.3924997e-03 4.5404971e-02 7.2941832e-05 2.5207126e-01 1.2750837e-03]]\n",
            "Predicted: cat\n"
          ]
        }
      ],
      "source": [
        "import json\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "\n",
        "# Read class names back from the JSON file\n",
        "with open(\"classes.json\", \"r\") as f:\n",
        "    class_names = json.load(f)\n",
        "\n",
        "# Load the model with the correct case 'CAT.keras'\n",
        "m = tf.keras.models.load_model(\"CAT.keras\")\n",
        "\n",
        "# Use a valid sample image from x_test\n",
        "sample_idx = 0\n",
        "x = np.expand_dims(x_test[sample_idx], axis=0)\n",
        "\n",
        "# Predict\n",
        "p = m.predict(x)\n",
        "print(p)\n",
        "print(f\"Predicted: {class_names[p.argmax()]}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "3a67e320",
      "metadata": {
        "id": "3a67e320"
      },
      "source": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.12.10"
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}