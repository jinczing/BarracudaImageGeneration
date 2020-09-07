using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class GauGANTester : MonoBehaviour
{
    public NNModel modelSource;
    public IWorker worker;
    public string name1;
    public string name2;
    public Image image;
    public int updateInterval;

    private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
    private Tensor output;
    private RenderTexture texture;
    private IOps ops;
    //private BurstCPUOps cops;
    private int count;
    private IEnumerator executor;

    // Start is called before the first frame update
    void Start()
    {
        ops = new PrecompiledComputeOps(ComputeShaderSingleton.Instance.kernels, ComputeShaderSingleton.Instance.referenceKernels);
        //ops = new BurstCPUOps();

        TextureLoader loader = new TextureLoader();

        print(TextureLoader.instance.semInput.shape);

        inputs[name1] = TextureLoader.instance.semInput;
        inputs[name2] = TextureLoader.instance.embedInput;

        var model = ModelLoader.Load(modelSource);

        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        //worker.Execute(inputs);
        //output = worker.PeekOutput();
        //output.PrintDataPart(100);
        //output = Normalize(output, ops);
        //texture = output.ToRenderTexture(0, 0, 1, 0, null);

        count = Time.frameCount;
    }

    // Update is called once per frame
    void Update()
    {
        if(Time.frameCount - count > updateInterval)
        {
            UpdateInput();
            count = Time.frameCount;

            // async execution
            //ExecuteInParts(worker, inputs, 5);
            //var output = executor = worker.ExecuteAsync(inputs);

            // sync execution
            float start = Time.time;
            worker.Execute(inputs);
            print(Time.time - start);
            var output = worker.PeekOutput();

            // normalize generated image
            output = Normalize(output, ops);

            // render
            texture = output.ToRenderTexture(0, 0, 1, 0, null);
        }
        //if(!executor.MoveNext())
        //{
        //    output = worker.PeekOutput();
        //    texture = output.ToRenderTexture(0, 0, 1, 0, null);
        //}
        image.canvasRenderer.SetTexture(texture);
    }

    private void OnDestroy()
    {
        worker.Dispose();
    }

    private Tensor ExecuteInParts(IWorker worker, Dictionary<string, Tensor> inputs, int syncLayers)
    {
        var executor = worker.ExecuteAsync(inputs);
        var iter = 0;
        bool hasMoreWork;

        do
        {
            hasMoreWork = executor.MoveNext();
            if (++iter % syncLayers == 0)
                worker.WaitForCompletion();
            
        } while (hasMoreWork);

        return worker.CopyOutput();
    }

    private void UpdateInput()
    {
        Tensor t = ops.RandomNormal(new TensorShape(1, 256, 512, 36), 0, 0.1f, 0);
        inputs[name1] = ops.Add(new Tensor[] { inputs[name1], t });
        t.Dispose();
    }
    
    private void PrintOutputShapes(Model model)
    {
        var inputShapes = new Dictionary<string, TensorShape>();
        foreach (var i in model.inputs)
        {
            inputShapes.Add(i.name, new TensorShape(i.shape));
        }
        IDictionary<string, TensorShape?> shapesByName;
        ModelAnalyzer.ListTemporaryTensorShapes(model, inputShapes, out shapesByName);
        print(shapesByName);
        foreach (var pair in shapesByName)
        {
            print($"{pair.Key} {pair.Value}");
        }
    }

    private Tensor Normalize(Tensor t, IOps ops)
    {
        Tensor min = Min(t, ops);
        t = ops.Sub(new Tensor[] { t, min });
        Tensor max = Max(t, ops);
        t = ops.Div(new Tensor[] { t, max });
        min.Dispose();
        max.Dispose();
        return t;
    }

    private Tensor Min(Tensor t,IOps ops)
    {
        var neg_t = ops.Neg(t);
        //neg_t.PrintDataPart(100);
        Tensor cmin = ops.GlobalMaxPool2D(neg_t).Flatten();
        //cmin.PrintDataPart(30);
        float min = 1000000000f;
        for (int i = 0; i < cmin.channels; ++i)
        {
            if (-cmin[0, i] < min)
                min = -cmin[0, i];
        }
        neg_t.Dispose();
        cmin.Dispose();
        return new Tensor(new int[] { 1 }, new float[] { min }, null);
    }

    private Tensor Max(Tensor t, IOps ops)
    {
        Tensor cmax = ops.GlobalMaxPool2D(t).Flatten();
        float max = -1000000000f;
        for (int i = 0; i < cmax.channels; ++i)
        {
            if (cmax[0, i] > max)
                max = cmax[0, i];
        }
        cmax.Dispose();
        return new Tensor(new int[] { 1 }, new float[] { max }, null);
    }
}
