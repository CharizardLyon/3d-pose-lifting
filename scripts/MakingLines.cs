using UnityEngine;

public class MakingLines : MonoBehaviour
{
    LineRenderer lineRenderer;

    public Transform origin;
    public Transform destination;

    void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.positionCount = 2; // Asegura que tenga dos puntos
        lineRenderer.startWidth = 0.05f;
        lineRenderer.endWidth = 0.05f;
    }

    // Update is called once per frame
    void Update()
    {
        lineRenderer.SetPosition(0, origin.position);
        lineRenderer.SetPosition(1, destination.position);
        
    }
}
