import onnxruntime as ort
from PIL import Image
from torchvision import transforms


# Input pre-processing for validation data
def preprocess_transform():
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform


def read_labels(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(line.strip())
    return out


if __name__ == "__main__":
    img_file = "cat.jpg"
    input_shape = [1, 3, 224, 224]
    input_image = Image.open(img_file)
    preprocess = preprocess_transform()
    input_data = preprocess(input_image).numpy().reshape(input_shape)

    in_name = "input"
    model_path = "mobilenetv2-10.onnx"
    shl_session = ort.InferenceSession(
        model_path,
        providers=["ShlExecutionProvider"],
        provider_options=[
            {
                # "debug_run_time": 1,
                # "debug_level": "CSINN_DEBUG_LEVEL_DEBUG",
                # "profiler_level": "CSINN_PROFILER_LEVEL_TIMER",
            },
        ],
    )

    output = shl_session.run(None, {in_name: input_data})[0].reshape(-1)
    result = reversed(output.argsort()[-5:])

    labels = read_labels("labels_map.txt")
    print(" ********** probability top5: **********")
    for r in result:
        print(f"{r}: prob {output[r]:.4f}, cls: {labels[r]}")

def infer_image(img_file):
    input_shape = [1, 3, 224, 224]
    input_image = Image.open(img_file)
    preprocess = preprocess_transform()
    input_data = preprocess(input_image).numpy().reshape(input_shape)

    in_name = "input"
    model_path = "mobilenetv2-10.onnx"
    shl_session = ort.InferenceSession(
        model_path,
        providers=["ShlExecutionProvider"],
        provider_options=[
            {
                # "debug_run_time": 1,
                # "debug_level": "CSINN_DEBUG_LEVEL_DEBUG",
                # "profiler_level": "CSINN_PROFILER_LEVEL_TIMER",
            },
        ],
    )

    output = shl_session.run(None, {in_name: input_data})[0].reshape(-1)
    result = reversed(output.argsort()[-5:])

    labels = read_labels("labels_map.txt")
    print(" ********** probability top5: **********")
    for r in result:
        print(f"{r}: prob {output[r]:.4f}, cls: {labels[r]}")
