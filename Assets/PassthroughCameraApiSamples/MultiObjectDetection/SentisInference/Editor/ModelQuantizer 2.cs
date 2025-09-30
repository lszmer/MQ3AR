using UnityEditor;
using UnityEngine;
using Unity.Sentis;

public static class ModelQuantizer
{
    [MenuItem("Sentis/Quantize Model")]
    public static void Quantize()
    {
        // Load ONNX
        var model = ModelLoader.Load("Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolo12n.onnx");

        // Quantize
        var quantizedModel = ModelOptimizer.Quantize(model, QuantizationType.Int8);

        // Serialize to StreamingAssets
        ModelLoader.SerializeToStreamingAssets(quantizedModel, "yolov12quantized.sentis");

        // Refresh assets so Unity detects the new file
        AssetDatabase.Refresh();

        Debug.Log("Quantized model saved to StreamingAssets as yolov12quantized.sentis");
    }
}