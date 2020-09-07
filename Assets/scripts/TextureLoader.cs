using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class TextureLoader : MonoBehaviour
{
    static public TextureLoader instance;


    public Tensor semInput;
    public Tensor embedInput;

    public Image edgeImage;

    private void Awake()
    {
        instance = this;

        PrecompiledComputeOps ops = new PrecompiledComputeOps(ComputeShaderSingleton.Instance.kernels, ComputeShaderSingleton.Instance.referenceKernels);
        BurstCPUOps cops = new BurstCPUOps();
       
        Texture2D seg_texture = Resources.Load<Texture2D>("Images/test_input");
        print(seg_texture.width);
        // seg_texture.Resize(256, 512);
        // seg_texture.Apply();
        var seg = new Tensor(seg_texture, 1);
        seg = ops.AvgPool2D(seg, new int[] { 1, 1 }, new int[] { 4, 4 }, new int[] { 0, 0, 0, 0 });
        print(seg.shape);
        seg.PrintDataPart(10);

        Texture2D edge_texture = Resources.Load<Texture2D>("Images/instance_input");
        // edge_texture.Resize(256, 512);
        // edge_texture.Apply();
        var edge = new Tensor(edge_texture, 1);
        edge = ops.AvgPool2D(edge, new int[] { 1, 1 }, new int[] { 4, 4 }, new int[] { 0, 0, 0, 0 });
        print(edge.shape);
        edge.PrintDataPart(100);

        seg = ops.Mul(new Tensor[] { seg, new Tensor(new int[1] { 1 }, new float[] { 255 }, null) });
        edge = ops.Mul(new Tensor[] { edge, new Tensor(new int[1] { 1 }, new float[] { 255 }, null) });
        seg.PrintDataPart(10);
        edge.PrintDataPart(10);
        var seg_input = new Tensor(1, 256, 512, 35);
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                for (int k = 0; k < 35; ++k)
                {
                    seg_input[0, i, j, (int)seg[0, i, j, 0]] = 1;
                }
            }
        }
        print("seg_input");
        seg_input.PrintDataPart(10);
        var edge_input = new Tensor(1, 256, 512, 1);
        print("edge_intput");
        edge_input.PrintDataPart(10);
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 1; j < 512; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i, j, 0] != edge[0, i, j - 1, 0])) ? 1.0f : 0.0f;
            }
        }
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 511; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i, j + 1, 0] != edge[0, i, j, 0])) ? 1.0f : 0.0f;
            }
        }
        for (int i = 1; i < 256; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i, j, 0] != edge[0, i - 1, j, 0])) ? 1.0f : 0.0f;
            }
        }
        for (int i = 0; i < 255; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i + 1, j, 0] != edge[0, i, j, 0])) ? 1.0f : 0.0f;
            }
        }
        var sem_input = cops.Concat(new Tensor[] { seg_input, edge_input }, 3);
        print(sem_input.shape);
        var embed_input = ops.AvgPool2D(sem_input, new int[] { 1, 1 }, new int[] { 64, 64 }, new int[] { 0, 0, 0, 0 });
        print(embed_input.shape);
        // edgeImage.material.mainTexture = edge_input.ToRenderTexture(0, 0, 1, 0, null);
        seg.Dispose();
        edge.Dispose();
        seg_input.Dispose();
        edge_input.Dispose();
        semInput = sem_input;
        embedInput = embed_input;
        // return new Tensor[] { sem_input, embed_input };
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    public Tensor[] LoadTexture()
    {
        PrecompiledComputeOps ops = new PrecompiledComputeOps(ComputeShaderSingleton.Instance.kernels, ComputeShaderSingleton.Instance.referenceKernels);
        BurstCPUOps cops = new BurstCPUOps();

        Texture2D seg_texture = Resources.Load<Texture2D>("Images/test_input");
        print(seg_texture.width);
        // seg_texture.Resize(256, 512);
        // seg_texture.Apply();
        var seg = new Tensor(seg_texture, 1);
        seg = ops.AvgPool2D(seg, new int[] { 1, 1 }, new int[] { 4, 4 }, new int[] { 0, 0, 0, 0});
        print(seg.shape);
        seg.PrintDataPart(10);

        Texture2D edge_texture = Resources.Load<Texture2D>("Images/instance_input");
        // edge_texture.Resize(256, 512);
        // edge_texture.Apply();
        var edge = new Tensor(edge_texture, 1);
        edge = ops.AvgPool2D(edge, new int[] { 1, 1 }, new int[] { 4, 4 }, new int[] { 0, 0, 0, 0 });
        print(edge.shape);
        edge.PrintDataPart(10);

        seg = ops.Mul(new Tensor[] { seg, new Tensor(new int[1] { 1 }, new float[] { 255 }, null) });
        edge = ops.Mul(new Tensor[] { edge, new Tensor(new int[1] { 1 }, new float[] { 255 }, null) });
        seg.PrintDataPart(10);
        edge.PrintDataPart(10);
        var seg_input = new Tensor(1, 256, 512, 35);
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                for (int k = 0; k < 35; ++k)
                {
                    if (k == seg[0, i, j, 0])
                    {
                        seg_input[0, i, j, k] = 1;
                    }
                    else
                    {
                        seg_input[0, i, j, k] = 0;
                    }
                }
            }
        }
        print("seg_input");
        seg_input.PrintDataPart(10);
        var edge_input = new Tensor(1, 256, 512, 1);
        print("edge_intput");
        edge_input.PrintDataPart(10);
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 1; j < 512; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i, j, 0] != edge[0, i, j - 1, 0])) ? 1.0f : 0.0f;
            }
        }
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 511; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i, j + 1, 0] != edge[0, i, j, 0])) ? 1.0f : 0.0f;
            }
        }
        for (int i = 1; i < 256; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i, j, 0] != edge[0, i - 1, j, 0])) ? 1.0f : 0.0f;
            }
        }
        for (int i = 0; i < 255; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                edge_input[0, i, j, 0] = ((edge_input[0, i, j, 0] == 1.0) | (edge[0, i + 1, j, 0] != edge[0, i, j, 0])) ? 1.0f : 0.0f;
            }
        }
        var sem_input = cops.Concat(new Tensor[] { seg_input, edge_input }, 3);
        print(sem_input.shape);
        var embed_input = ops.AvgPool2D(sem_input, new int[] { 1, 1 }, new int[] { 64, 64 }, new int[] { 0, 0, 0, 0 });
        print(embed_input.shape);
        seg.Dispose();
        edge.Dispose();
        seg_input.Dispose();
        edge_input.Dispose();
        return new Tensor[] { sem_input, embed_input };
    }

    private void Fill(Tensor tensor, float value)
    {
        for(int i=0; i!=tensor.batch; ++i)
        {
            for(int j=0; j!=tensor.width; ++j)
            {
                for(int k=0; k!=tensor.height; ++k)
                {
                    for(int l=0; l!=tensor.channels; ++l)
                    {
                        tensor[i, j, k, l] = value;
                    }
                }
            }
        }
    }
}
