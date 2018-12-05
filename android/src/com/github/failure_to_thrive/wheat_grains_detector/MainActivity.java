package com.github.failure_to_thrive.wheat_grains_detector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.content.res.AssetManager;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "WGD";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Net net;
//    private String path;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        try {
            net = Dnn.readNetFromTensorflow(Extract("frozen_inference_graph.pb"), Extract("config.pbtxt"));
            Log.i(TAG, "Network loaded successfully");
        } catch (IOException e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
            finish();
        }
/*
        File sd = new File(android.os.Environment.getExternalStorageDirectory(), "WGD_frames/");
        if (!sd.exists())
            sd.mkdir();
        path = sd.getPath();
*/
    }

    // Extract a file from assets to a storage and return a path.
    private String Extract(String filename) throws IOException {
        AssetManager assetManager = getAssets();
        BufferedInputStream inputStream = new BufferedInputStream(assetManager.open(filename));
        byte[] data = new byte[inputStream.available()];
        inputStream.read(data);
        inputStream.close();
        File outFile = new File(getFilesDir(), filename);
        FileOutputStream outputStream = new FileOutputStream(outFile);
        outputStream.write(data);
        outputStream.close();
        return outFile.getAbsolutePath();
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        int cols = frame.cols();
        int rows = frame.rows();

        // Pad a frame to make it square. Shrinking with a loss of aspect ratio so commonly applying everywhere is a bad idea.
        int largest = Math.max(cols, rows);
        Mat square = new Mat();
        Core.copyMakeBorder(frame, square, 0, largest - rows, 0, largest - cols, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

        Mat blob = Dnn.blobFromImage(square, 1,
                new Size(300, 300),
                new Scalar(0, 0, 0), false, false);
        net.setInput(blob);
        Mat detections = net.forward();

        detections = detections.reshape(0, (int)detections.total() / 7);
        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > 0.3) {
                int classId = (int)detections.get(i, 1)[0];
                int left = (int)(detections.get(i, 3)[0] * cols);
                int top = (int)(detections.get(i, 4)[0] * rows);
                int right = (int)(detections.get(i, 5)[0] * cols);
                int bottom = (int)(detections.get(i, 6)[0] * rows);
                // Bring coordinates back to the original frame.
                if (cols > rows) {
                    top *= (double)cols/rows;
                    bottom *= (double)cols/rows;
                }
                else {
                    left *= (double)rows/cols;
                    right *= (double)rows/cols;
                }

                Scalar color;
                switch (classId) {
                    case 2:
                    case 3:
                        color = confidence > 0.6 ? new Scalar(255, 0, 0) : new Scalar(255, 255, 0);
                        break;
                    default:
                        color = new Scalar(127, 127, 127);
                }
                // Draw rectangle around detected object.
                Imgproc.rectangle(frame, new Point(left, top),
                        new Point(right, bottom),
                        color, 2);
/*
                // Print class and confidence.
                String label = String.format("%d %.3f", classId, confidence);
                int[] baseline = new int[1];
                Size labelSize = Imgproc.getTextSize(label, Core.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseline);
                Imgproc.rectangle(frame, new Point(left, top),
                        new Point(left + labelSize.width, top + labelSize.height + baseline[0]),
                        color, Core.FILLED);
                Imgproc.putText(frame, label, new Point(left, top + labelSize.height + baseline[0]/2),
                        Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));
*/
             }
        }
/*
        Mat save = new Mat();
        Imgproc.cvtColor(frame, save, Imgproc.COLOR_RGB2BGR);
        org.opencv.imgcodecs.Imgcodecs.imwrite(path + String.format("/%tj_%1$tH%1$tM%1$tS%1$tL.jpg", new java.util.Date()), save);
*/
        return frame;
    }
}
