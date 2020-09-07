using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

public class OnnxTester : MonoBehaviour
{
    public NNModel modelSource;
    public string inputName1;
    public string inputName2;
    public string outputName1;
    public Material targetMaterial;
    public Image image;
    public float pf;
    public float walkStrength;

    private float interval;
    private IWorker worker;
    private int counter;
    private Dictionary<string, Tensor> inputs;
    private float[] cond;
    private float[] noise;
    private Tensor output;
    private RenderTexture targetTexture;

    // Start is called before the first frame update
    void Start()
    {
        inputs = new Dictionary<string, Tensor>();
        cond = new float[18];
        for (int i = 0; i < 18; ++i)
        {
            cond[i] = Random.Range(0, 0.3f);
        }
        inputs[inputName2] = new Tensor(1, 1, 1, 18, cond);
        noise = new float[100];
        for (int i = 0; i < 100; ++i)
        {
            noise[i] = Random.Range(-1f, 1f);
        }
        inputs[inputName1] = new Tensor(1, 1, 1, 100, noise);
        var model = ModelLoader.Load(modelSource);
        worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model);
        print("Update Per Frame: " + pf);
        counter = Time.frameCount;
    }

    // Update is called once per frame
    void Update()
    {
        if(Time.frameCount - counter >= pf)
        {
            counter = Time.frameCount;
            for (int i = 0; i < 100; ++i)
            {
                inputs[inputName1][0,0,0,i] += Random.Range(-walkStrength, walkStrength);
            }
            worker.Execute(inputs);
            output = worker.PeekOutput();
            targetTexture = output.ToRenderTexture(0, 0, 1, 0, null);
            image.canvasRenderer.SetTexture(targetTexture);
            output.Dispose();
        }
    }
}
