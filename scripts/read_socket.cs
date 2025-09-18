using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class TCPClientUnity : MonoBehaviour
{
    public string serverIP = "127.0.0.1";
    public int port = 65432;

    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;

    void Start()
    {
        ConnectToServer();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient(serverIP, port);
            stream = client.GetStream();
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();
            Debug.Log("Connected to Python server");
        }
        catch (Exception e)
        {
            Debug.LogError("Socket connection error: " + e.Message);
        }
    }

    void ReceiveData()
    {
        while (true)
        {
            try
            {
                // Read length prefix (4 bytes)
                byte[] lengthBytes = new byte[4];
                int read = stream.Read(lengthBytes, 0, 4);
                if (read == 0) break;
                int messageLength = BitConverter.ToInt32(lengthBytes, 0);

                // Read the JSON message data
                byte[] dataBytes = new byte[messageLength];
                int received = 0;
                while (received < messageLength)
                {
                    int bytesRead = stream.Read(dataBytes, received, messageLength - received);
                    if (bytesRead == 0) break;
                    received += bytesRead;
                }

                string jsonString = Encoding.UTF8.GetString(dataBytes);
                // Process JSON string (example: print in console)
                Debug.Log("Received Pose Data: " + jsonString);

                // You can parse JSON to float[][] using Unity's JsonUtility or third-party libs
            }
            catch (Exception e)
            {
                Debug.LogError("ReceiveData error: " + e.Message);
                break;
            }
        }
    }

    private void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
}
