{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPjNUsbhB2MrmOBz3MzgSoV",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/lalmankamal89-hub/IMAGE-CLASSIFICATION_Lalman/blob/main/Image_.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit tensorflow pillow\n",
        "!streamlit run app.py\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JOrknuDUyimv",
        "outputId": "f24d2485-72c2-4a8d-8dda-0d356352a5bb"
      },
      "execution_count": none,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.13/dist-packages (1.66.1)\n",
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.13/dist-packages (2.21.0)\n",
            "Requirement already satisfied: pillow in /usr/local/lib/python3.13/dist-packages (11.3.0)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=5.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: numpy<3,>=1.25.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.1.3)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.13/dist-packages (from streamlit) (26.3)\n",
            "Requirement already satisfied: pandas<4,>=1.5.3 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.3)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.9.3)\n",
            "Requirement already satisfied: protobuf<8,>=5.26.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.33.6)\n",
            "Requirement already satisfied: pyarrow!=25.0.0,<27,>=10.0.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (23.0.1)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.10.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.16.0)\n",
            "Requirement already satisfied: starlette<2,>=0.46.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (1.10.0)\n",
            "Requirement already satisfied: uvicorn<1,>=0.30.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.54.0)\n",
            "Requirement already satisfied: anyio<5,>=4.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.15.1)\n",
            "Requirement already satisfied: python-multipart<1,>=0.0.10 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.0.32)\n",
            "Requirement already satisfied: websockets<18,>=12.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (15.0.1)\n",
            "Requirement already satisfied: itsdangerous<3,>=2.1.2 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (4.26.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2.28.0)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.13/dist-packages (from anyio<5,>=4.0.0->streamlit) (3.20)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2026.5)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (3.4.9)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2026.10.23)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.13/dist-packages (from uvicorn<1,>=0.30.0->streamlit) (0.16.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (2.0.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=25.9.23 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (25.12.19)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (0.7.0)\n",
            "Requirement already satisfied: google_pasta>=0.1.1 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: opt_einsum>=2.3.2 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.4.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.13/dist-packages (from tensorflow) (80.10.2)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (1.17.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.3.0)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (2.5.1)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (1.85.0)\n",
            "Requirement already satisfied: keras>=3.12.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.13.2)\n",
            "Requirement already satisfied: h5py<3.15.0,>=3.11.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.14.0)\n",
            "Requirement already satisfied: ml_dtypes<1.0.0,>=0.5.1 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.13/dist-packages (from astunparse>=1.6.0->tensorflow) (0.48.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.13/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.0.4)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (26.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2026.9.1)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.13/dist-packages (from keras>=3.12.0->tensorflow) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.13/dist-packages (from keras>=3.12.0->tensorflow) (0.1.0)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.13/dist-packages (from keras>=3.12.0->tensorflow) (0.20.0)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.13/dist-packages (from rich->keras>=3.12.0->tensorflow) (4.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.13/dist-packages (from rich->keras>=3.12.0->tensorflow) (2.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.13/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.12.0->tensorflow) (0.1.2)\n",
            "Usage: streamlit run [OPTIONS] [TARGET] [ARGS]...\n",
            "Try 'streamlit run --help' for help.\n",
            "\n",
            "Error: Invalid value: File does not exist: app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d65dbca4",
        "outputId": "99409558-3803-4936-fbb4-48ca6103faaa"
      },
      "source": [
        "!pip install streamlit"
      ],
      "execution_count": none,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.13/dist-packages (1.66.1)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=5.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: numpy<3,>=1.25.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.1.3)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.13/dist-packages (from streamlit) (26.3)\n",
            "Requirement already satisfied: pandas<4,>=1.5.3 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.3)\n",
            "Requirement already satisfied: pillow<14,>=9.2.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (11.3.0)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.9.3)\n",
            "Requirement already satisfied: protobuf<8,>=5.26.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.33.6)\n",
            "Requirement already satisfied: pyarrow!=25.0.0,<27,>=10.0.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (23.0.1)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.10.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.16.0)\n",
            "Requirement already satisfied: starlette<2,>=0.46.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (1.10.0)\n",
            "Requirement already satisfied: uvicorn<1,>=0.30.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.54.0)\n",
            "Requirement already satisfied: anyio<5,>=4.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.15.1)\n",
            "Requirement already satisfied: python-multipart<1,>=0.0.10 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.0.32)\n",
            "Requirement already satisfied: websockets<18,>=12.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (15.0.1)\n",
            "Requirement already satisfied: itsdangerous<3,>=2.1.2 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (4.26.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2.28.0)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.13/dist-packages (from anyio<5,>=4.0.0->streamlit) (3.20)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2026.5)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (3.4.9)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2026.10.23)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.13/dist-packages (from uvicorn<1,>=0.30.0->streamlit) (0.16.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.13/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.0.4)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (26.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2026.9.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.13/dist-packages (from python-dateutil>=2.8.2->pandas<4,>=1.5.3->streamlit) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": none,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EPnVPtMUwUBZ",
        "outputId": "ffd18376-d2ba-4a19-a828-436459d88a31"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import json\n",
        "\n",
        "import numpy as np\n",
        "import streamlit as st\n",
        "import tensorflow as tf\n",
        "from PIL import Image\n",
        "\n",
        "@st.cache_resource\n",
        "def load_model_and_classes():\n",
        "    model = tf.keras.models.load_model(\"/content/drive/MyDrive/CAT.keras\")\n",
        "    with open(\"/content/drive/MyDrive/classes.json\", \"r\") as f:\n",
        "        class_names = json.load(f)\n",
        "    return model, class_names\n",
        "\n",
        "model, class_names = load_model_and_classes()\n",
        "\n",
        "st.title(\"Image Classifier\")\n",
        "\n",
        "uploaded_file = st.file_uploader(\n",
        "    \"Upload an image\",\n",
        "    type=[\"jpg\", \"jpeg\", \"png\"],\n",
        ")\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    image = Image.open(uploaded_file).convert(\"RGB\")\n",
        "    st.image(image, caption=\"Uploaded image\", use_container_width=True)\n",
        "\n",
        "    # Resize to the model's expected input size.\n",
        "    height, width = model.input_shape[1:3]\n",
        "    if height is None or width is None:\n",
        "        st.error(\"The model needs a fixed image height and width.\")\n",
        "    else:\n",
        "        image = image.resize((width, height))\n",
        "        x = np.asarray(image, dtype=np.float32)\n",
        "        x = np.expand_dims(x, axis=0)\n",
        "\n",
        "        # Apply the same preprocessing used when training the model.\n",
        "        # For example, if training used pixel values in [0, 1], uncomment:\n",
        "        # x = x / 255.0\n",
        "\n",
        "        predictions = model.predict(x, verbose=0)[0]\n",
        "        predicted_index = int(np.argmax(predictions))\n",
        "\n",
        "        st.write(f\"**Predicted:** {class_names[predicted_index]}\")\n",
        "        st.write(f\"**Confidence:** {predictions[predicted_index]:.2%}\")"
      ]
    }
  ]
}
