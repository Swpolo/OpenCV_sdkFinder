package paul.opencv_sdkfinder;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.CvType.CV_8U;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2, SurfaceView.OnTouchListener {

    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    public SurfaceView.OnTouchListener mOpenCvCameraViewOnTouchListener;

    private Mat         m_gui_area;
    private Mat         m_draw;
    private Mat         m_gray;
    private Mat         m_AT_pending;
    private Mat         m_AT_finished;
    private Mat         m_sdk;

    private AT_findSdk  at_findSdk;
    private boolean     at_finished;

    private boolean     sdkFound;

    private enum        e_gui{SHOW_RECTANGLE, SHOW_SDK}
    private boolean[]   b_gui = new boolean[2];

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
        System.loadLibrary("opencv_java3");
    }

    private int cameraPermission(){
        int hasPermission = ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.CAMERA);
        if (hasPermission != PackageManager.PERMISSION_GRANTED) {

            if (!ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,
                    Manifest.permission.CAMERA)) {

                // TODO : Ajouter un dialogue

            }
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.CAMERA},
                    1);
            hasPermission = ContextCompat.checkSelfPermission(MainActivity.this,
                    Manifest.permission.CAMERA);
        }

        return hasPermission;
    }

    private int writeExternalStoragePermission(){
        int hasPermission = ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (hasPermission != PackageManager.PERMISSION_GRANTED) {

            if(!ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)) {

                // TODO : Ajouter un dialogue

            }

            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    1);

            hasPermission = ContextCompat.checkSelfPermission(MainActivity.this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        return hasPermission;
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    cameraPermission();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.MainActivity_SurfaceView);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        mOpenCvCameraView.setOnTouchListener(mOpenCvCameraViewOnTouchListener);
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
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onCameraViewStarted(int width, int height) {
        m_gui_area       = new Mat(height, width, CvType.CV_8UC4);
        createGuiSdkCpp(m_gui_area.getNativeObjAddr());

        m_draw          = new Mat(height, width, CvType.CV_8UC4);

        m_gray          = new Mat(height, width, CvType.CV_8UC1);
        m_sdk           = new Mat(height, width, CvType.CV_8UC1);

        m_AT_finished   = new Mat(height, width, CvType.CV_8UC4);
        m_AT_pending    = new Mat(height, width, CvType.CV_8UC4);

        at_findSdk      = new AT_findSdk();

        for ( int i = 0; i < b_gui.length; i++){
            b_gui[i] = false;
        }

    }

    public void onCameraViewStopped() {
        m_gui_area.release();

        m_draw.release();

        m_gray.release();
        m_sdk.release();

        m_AT_finished.release();
        m_AT_pending.release();

    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        findSdk(inputFrame.gray());

        inputFrame.rgba().copyTo(m_draw);

//        if(!b_gui[e_gui.SHOW_SDK.ordinal()]) {
            addWeightedCpp(m_draw.getNativeObjAddr(), m_gui_area.getNativeObjAddr(), 1.0, 0.7);
//        }
//
//        if (at_finished) {
//            if(b_gui[e_gui.SHOW_RECTANGLE.ordinal()]) {
                addWeightedCpp(m_draw.getNativeObjAddr(), m_AT_finished.getNativeObjAddr(), 1.0, 1.0);
//            }
//            else if(b_gui[e_gui.SHOW_SDK.ordinal()] && sdkFound){
//                addWeightedCpp(m_draw.getNativeObjAddr(), m_sdk.getNativeObjAddr(), 1.0, 1.0);
//            }
//        }
        Scalar bt;
        if(b_gui[e_gui.SHOW_RECTANGLE.ordinal()]){
            bt = new Scalar(255,0,0);
        }
        else if(b_gui[e_gui.SHOW_SDK.ordinal()]){
            bt = new Scalar(0,255,0);
        }
        else{
            bt = new Scalar(0,0,255);
        }

        Imgproc.rectangle(m_draw, new Point(10,10), new Point(20,20), bt,-1);
        return m_draw;
    }

    private void findSdk(Mat gray){
        if(at_findSdk.getStatus() == AsyncTask.Status.PENDING){
            gray.copyTo(m_gray);
            at_findSdk.execute(m_gray, m_AT_pending);
        }
        else if (at_findSdk.getStatus() == AsyncTask.Status.FINISHED) {
            m_gray.copyTo(m_sdk);
            m_AT_pending.copyTo(m_AT_finished);
            Mat.zeros(m_AT_pending.rows(),m_AT_pending.cols(), m_AT_pending.type()).copyTo(m_AT_pending);
            at_finished = true;
            sdkFound = at_findSdk.getSdkFound();
            at_findSdk = new AT_findSdk();
        }
    }

    private class AT_findSdk extends AsyncTask<Mat, Void, Void> {
        private boolean at_sdkFound;

        protected Void doInBackground(Mat... params){
            at_sdkFound = findSdkCpp(params[0].getNativeObjAddr(), params[1].getNativeObjAddr());
            return null;
        }

        protected boolean getSdkFound() {
            return at_sdkFound;
        }
    }

    @Override
    public boolean onTouch( View v, MotionEvent event) {

        switch(event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                if (!b_gui[e_gui.SHOW_RECTANGLE.ordinal()] && !b_gui[e_gui.SHOW_SDK.ordinal()]) {
                    b_gui[e_gui.SHOW_RECTANGLE.ordinal()] = false;
                    b_gui[e_gui.SHOW_SDK.ordinal()] = true;
                } else if (b_gui[e_gui.SHOW_RECTANGLE.ordinal()] && !b_gui[e_gui.SHOW_SDK.ordinal()]) {
                    b_gui[e_gui.SHOW_RECTANGLE.ordinal()] = false;
                    b_gui[e_gui.SHOW_SDK.ordinal()] = false;
                } else {
                    b_gui[e_gui.SHOW_RECTANGLE.ordinal()] = true;
                    b_gui[e_gui.SHOW_SDK.ordinal()] = false;
                }

                Log.d("POLO:onTouch", "ON TOUCHE !!!");
                break;
            default:
                Log.d("POLO:onTouch", "ON TOUCHE QUEDAL!!!");
                break;
        }

        return true;
    }

    public native void createGuiSdkCpp(long matAddr);

    public native boolean findSdkCpp(long grayAddr, long outAddr);

    public native void addWeightedCpp(long m1Addr, long m2Addr, double alpha, double beta);
}
