using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class Lab : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Tensor t = new Tensor(1, 5);
        t[0, 2] = -1;
        BurstCPUOps cops = new BurstCPUOps();
        cops.Min(new Tensor[] { t }).PrintDataPart(5);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
