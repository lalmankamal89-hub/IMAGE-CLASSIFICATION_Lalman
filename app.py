{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPFOSBel6V2wPRLVR3mZawz",
      "include_colab_link": true
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
        "<a href=\"https://colab.research.google.com/github/lalmankamal89-hub/IMAGE-CLASSIFICATION_Lalman/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n"
      ],
      "metadata": {
        "id": "WLX0e7yd4yi5"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit tensorflow numpy matplotlib"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "le7pkMpH04V5",
        "outputId": "652385bc-dc8b-4075-8972-8044316f7410"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.66.1-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.13/dist-packages (2.21.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.13/dist-packages (2.4.6)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.13/dist-packages (3.10.0)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=5.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.13/dist-packages (from streamlit) (26.3)\n",
            "Requirement already satisfied: pandas<4,>=1.5.3 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.3)\n",
            "Requirement already satisfied: pillow<14,>=9.2.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (11.3.0)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.3-py2.py3-none-any.whl.metadata (4.2 kB)\n",
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
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2026.6)\n",
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
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.13/dist-packages (from matplotlib) (1.4.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.13/dist-packages (from matplotlib) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.13/dist-packages (from matplotlib) (4.68.0)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.13/dist-packages (from matplotlib) (1.5.1)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.13/dist-packages (from matplotlib) (3.3.3)\n",
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
            "Downloading streamlit-1.66.1-py3-none-any.whl (10.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.4/10.4 MB\u001b[0m \u001b[31m19.6 MB/s\u001b[0m  \u001b[33m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.3-py2.py3-none-any.whl (11.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.4/11.4 MB\u001b[0m \u001b[31m51.1 MB/s\u001b[0m  \u001b[33m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2/2\u001b[0m [streamlit]\n",
            "\u001b[1A\u001b[2KSuccessfully installed pydeck-0.9.3 streamlit-1.66.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dxgZezNi19b7",
        "outputId": "2b7e57d0-11ca-4b0e-83e6-b5322ba3df40"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
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
      "source": [
        "import streamlit as st\n",
        "\n",
        "st.title(\"Image Classification\")\n",
        "\n",
        "st.write(\"Application is running successfully!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SFtFB_Zn47SZ",
        "outputId": "c08cb6f4-323a-42a1-bd50-7090ab890a01"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2026-11-03 14:50:21.397 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:50:21.862 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.13/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2026-11-03 14:50:21.863 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:50:21.865 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:50:21.867 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:50:21.870 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:50:21.870 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import numpy as np\n",
        "import os\n",
        "from PIL import Image\n",
        "from tensorflow import keras\n",
        "\n",
        "# --------------------------------\n",
        "# Page Configuration\n",
        "# --------------------------------\n",
        "st.set_page_config(\n",
        "    page_title=\"Image Classification\",\n",
        "    page_icon=\"🖼️\",\n",
        "    layout=\"centered\"\n",
        ")\n",
        "\n",
        "st.title(\"🖼️ Image Classification using CNN\")\n",
        "st.write(\"Upload an image to predict its category.\")\n",
        "\n",
        "# --------------------------------\n",
        "# Load Trained Keras Model Safely\n",
        "# --------------------------------\n",
        "@st.cache_resource\n",
        "def load_model():\n",
        "    model_path = \"my_model.keras\"\n",
        "    if not os.path.exists(model_path):\n",
        "        return None\n",
        "    return keras.models.load_model(model_path)\n",
        "\n",
        "model = load_model()\n",
        "\n",
        "# Ensure variables from notebook exist or provide fallbacks\n",
        "try:\n",
        "    class_names = classification\n",
        "except NameError:\n",
        "    class_names = [\"Class 0\", \"Class 1\", \"Class 2\", \"Class 3\", \"Class 4\", \"Class 5\", \"Class 6\", \"Class 7\", \"Class 8\", \"Class 9\"]\n",
        "\n",
        "if model is None:\n",
        "    st.error(\"⚠️ 'my_model.keras' was not found in the workspace directory. Please train your model and save it using `model.save('my_model.keras')` first!\")\n",
        "else:\n",
        "    # --------------------------------\n",
        "    # Image Upload\n",
        "    # --------------------------------\n",
        "    uploaded_file = st.file_uploader(\n",
        "        \"Upload an Image\",\n",
        "        type=[\"jpg\", \"jpeg\", \"png\"]\n",
        "    )\n",
        "\n",
        "    if uploaded_file is not None:\n",
        "        image = Image.open(uploaded_file).convert(\"RGB\")\n",
        "        st.image(image, caption=\"Uploaded Image\", use_container_width=True)\n",
        "\n",
        "        # --------------------------------\n",
        "        # Image Preprocessing\n",
        "        # --------------------------------\n",
        "        img_size = (32, 32)  # Adjust depending on model input size\n",
        "        image = image.resize(img_size)\n",
        "        x = np.array(image)\n",
        "        x = x.astype(\"float32\") / 255.0\n",
        "        x = np.expand_dims(x, axis=0)\n",
        "\n",
        "        # --------------------------------\n",
        "        # Prediction\n",
        "        # --------------------------------\n",
        "        predictions = model.predict(x, verbose=0)\n",
        "        predicted_class = np.argmax(predictions[0])\n",
        "        confidence = float(np.max(predictions[0])) * 100\n",
        "\n",
        "        # --------------------------------\n",
        "        # Display Results\n",
        "        # --------------------------------\n",
        "        st.subheader(\"Prediction Results\")\n",
        "        if predicted_class < len(class_names):\n",
        "            st.success(f\"Predicted Class: {class_names[predicted_class]}\")\n",
        "        else:\n",
        "            st.success(f\"Predicted Class ID: {predicted_class}\")\n",
        "\n",
        "        st.metric(\"Confidence\", f\"{confidence:.2f}%\")\n",
        "\n",
        "        # Prediction probabilities\n",
        "        st.subheader(\"Prediction Probabilities\")\n",
        "        for i, name in enumerate(class_names[:len(predictions[0])]):\n",
        "            probability = float(predictions[0][i]) * 100\n",
        "            st.write(f\"{name}: {probability:.2f}%\")\n",
        "            st.progress(min(probability / 100, 1.0))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R0tm5-B5BsoA",
        "outputId": "9501ed62-6816-4c17-e749-5755f5d172ee"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ]
    }
  ]
}