using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class HandPoseRig : MonoBehaviour
{
    [Header("Networking")]
    public string ip = "127.0.0.1";
    public int port = 5005;

    [Header("Rig (21 GameObjects en orden)")]
    public GameObject[] handpoints; // wrist=0, luego dedos…

    [Header("Transform options")]
    public bool useLocalPosition = true; // true: localPosition, false: position
    public float scale = 0.01f;          // ajusta tamaño
    public Vector3 offset = Vector3.zero;
    [Tooltip("Cambia ejes si tu coord. de Python no coincide con Unity")]
    public AxisMap axisMap = AxisMap.XYZ;

    [Header("Debug")]
    public bool logEveryMessage = false;

    // === Internos ===
    private TcpListener server;
    private Thread serverThread;
    private readonly object poseLock = new object();
    private List<Vector3> latestPose = null;

    // Mapas de ejes comunes (edita si necesitas)
    public enum AxisMap { XYZ, XZY, YXZ, YZX, ZXY, ZYX, Custom }
    public Vector3 customAxisSign = new Vector3(1, 1, 1); // si usas Custom, signos por eje

    void Start()
    {
        if (handpoints == null || handpoints.Length != 21)
            Debug.LogWarning("Asigna 21 handpoints en el inspector.");

        serverThread = new Thread(ListenForData);
        serverThread.IsBackground = true;
        serverThread.Start();
    }

    void Update()
    {
        List<Vector3> snapshot = null;
        lock (poseLock)
        {
            if (latestPose != null)
            {
                snapshot = latestPose;
                latestPose = null; // consumir
            }
        }

        if (snapshot != null && handpoints != null && handpoints.Length == 21)
        {
            for (int i = 0; i < 21; i++)
            {
                Vector3 v = ApplyAxisMap(snapshot[i]) * scale + offset;
                if (useLocalPosition)
                    handpoints[i].transform.localPosition = v;
                else
                    handpoints[i].transform.position = v;
            }

            if (logEveryMessage)
                Debug.Log($"Pose recibida: {snapshot.Count} joints. J0={snapshot[0]}");
        }
    }

    void OnApplicationQuit()
    {
        try { server?.Stop(); } catch { }
        try { serverThread?.Abort(); } catch { }
    }

    // === Red ===
    private void ListenForData()
    {
        try
        {
            server = new TcpListener(IPAddress.Parse(ip), port);
            server.Start();
            Debug.Log($"Servidor iniciado en {ip}:{port}");

            while (true)
            {
                using (TcpClient client = server.AcceptTcpClient())
                using (NetworkStream stream = client.GetStream())
                {
                    Debug.Log("Cliente conectado.");

                    while (true)
                    {
                        // 1) longitud (4 bytes big-endian)
                        byte[] lenBuf = ReadExact(stream, 4);
                        if (lenBuf == null) break;

                        int msgLength = (lenBuf[0] << 24) |
                                        (lenBuf[1] << 16) |
                                        (lenBuf[2] << 8) |
                                        lenBuf[3];

                        // 2) cuerpo JSON
                        byte[] msgBuf = ReadExact(stream, msgLength);
                        if (msgBuf == null) break;

                        string json = Encoding.UTF8.GetString(msgBuf);

                        // 3) parsear [[x,y,z],...]
                        var pose = ParsePoseToVector3(json);

                        // 4) publicar al hilo principal
                        lock (poseLock)
                        {
                            latestPose = pose;
                        }
                    }
                }
            }
        }
        catch (SocketException e)
        {
            Debug.Log("SocketException: " + e);
        }
        catch (Exception ex)
        {
            Debug.Log("Exception: " + ex);
        }
    }

    private static byte[] ReadExact(NetworkStream stream, int count)
    {
        byte[] buffer = new byte[count];
        int offset = 0;
        while (offset < count)
        {
            int read = stream.Read(buffer, offset, count - offset);
            if (read == 0) return null; // conexión cerrada
            offset += read;
        }
        return buffer;
    }

    // === Parser JSON simple para [[x,y,z], ...] ===
    private static List<Vector3> ParsePoseToVector3(string json)
    {
        var result = new List<Vector3>(21);

        string clean = json.Trim();
        if (clean.Length < 5) return result;

        // quitar corchetes exteriores
        if (clean.StartsWith("[")) clean = clean.Substring(1);
        if (clean.EndsWith("]")) clean = clean.Substring(0, clean.Length - 1);

        // dividir por "],"
        string[] triplets = clean.Split(new string[] { "]," }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var t in triplets)
        {
            string triplet = t.Replace("[", "").Replace("]", "").Trim();
            string[] comps = triplet.Split(',');
            if (comps.Length < 3) continue;

            float x = float.Parse(comps[0], System.Globalization.CultureInfo.InvariantCulture);
            float y = float.Parse(comps[1], System.Globalization.CultureInfo.InvariantCulture);
            float z = float.Parse(comps[2], System.Globalization.CultureInfo.InvariantCulture);

            result.Add(new Vector3(x, y, z));
        }

        return result;
    }

    // === Mapeo de ejes (ajusta si tu Python no coincide con Unity) ===
    private Vector3 ApplyAxisMap(Vector3 v)
    {
        switch (axisMap)
        {
            case AxisMap.XZY: return new Vector3( v.x, v.z, v.y);
            case AxisMap.YXZ: return new Vector3( v.y, v.x, v.z);
            case AxisMap.YZX: return new Vector3( v.y, v.z, v.x);
            case AxisMap.ZXY: return new Vector3( v.z, v.x, v.y);
            case AxisMap.ZYX: return new Vector3( v.z, v.y, v.x);
            case AxisMap.Custom: return new Vector3(v.x * customAxisSign.x, v.y * customAxisSign.y, v.z * customAxisSign.z);
            default:            return v; // XYZ
        }
    }
}
