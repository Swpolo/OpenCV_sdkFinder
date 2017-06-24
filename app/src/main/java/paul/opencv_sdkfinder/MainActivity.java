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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

    enum State{
        seekingSDK,
        showSDK,
        stopCamera
    }

    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    public SurfaceView.OnTouchListener mOpenCvCameraViewOnTouchListener;


    private Rect        r_detectArea;
    private Mat         m_lastOutput;
    private Mat         m_draw;
    private Mat         m_gray;
    private Mat         m_sdk;

    private Mat         m_gui_area;
    private Mat         m_sub_draw;
    private Mat         m_sub_gray;
    private Mat         m_AT_pending;
    private Mat         m_AT_finished;

    private AT_findSdk  at_findSdk;
    private boolean     at_finished;

    private boolean     sdkFound;
    private boolean     outputChoice;

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
        Log.d("POLO_D", "START");
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.MainActivity_SurfaceView);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

//        mOpenCvCameraView.setMaxFrameSize(1000, 1000);

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
        Log.d("POLO_D", "onCameraViewStarted - A");
        // Screen sized mat
        m_lastOutput    = new Mat(height, width, CvType.CV_8UC4);

        m_draw          = new Mat(height, width, CvType.CV_8UC4);

        m_gray          = new Mat(height, width, CvType.CV_8UC1);


        // area detection size mat
        calculateDetectArea(width, height);
        Log.d("POLO_D", "onCameraViewStarted - 1");
        m_sub_draw      = new Mat(m_draw, r_detectArea);
        m_sub_gray      = new Mat(m_gray, r_detectArea);
        Log.d("POLO_D", "onCameraViewStarted - 2");

        m_gui_area      = new Mat(r_detectArea.height, r_detectArea.width, CvType.CV_8UC4);
        Log.d("POLO_D", "onCameraViewStarted - 3");
        createGuiSdk();

        m_AT_finished   = new Mat(r_detectArea.height, r_detectArea.width, CvType.CV_8UC4);
        m_AT_pending    = new Mat(r_detectArea.height, r_detectArea.width, CvType.CV_8UC4);
        m_sdk           = new Mat(r_detectArea.height, r_detectArea.width, CvType.CV_8UC4);

        at_findSdk      = new AT_findSdk();


        sdkFound        = false;
        outputChoice    = false;

        Log.d("POLO_D", "onCameraViewStarted - B");
    }

    public void onCameraViewStopped() {
        Log.d("POLO_D", "onCameraViewStopped - A");
        m_lastOutput.release();

        m_gui_area.release();

        m_draw.release();

        m_gray.release();
        m_sdk.release();

        m_AT_finished.release();
        m_AT_pending.release();

        Log.d("POLO_D", "onCameraViewStopped - B");
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (sdkFound){
            return m_lastOutput;
        }

        findSdk(inputFrame.gray());

        inputFrame.rgba().copyTo(m_draw);

        addWeightedCpp(m_sub_draw.getNativeObjAddr(), m_gui_area.getNativeObjAddr(), 1.0, 0.7);

        addWeightedCpp(m_sub_draw.getNativeObjAddr(), m_AT_finished.getNativeObjAddr(), 1.0, 1.0);

        m_draw.copyTo(m_lastOutput);
        return m_draw;
    }

    private void findSdk(Mat gray){
        if( !sdkFound ) {
            if (at_findSdk.getStatus() == AsyncTask.Status.PENDING) {
                gray.copyTo(m_gray);
                at_findSdk.setResultChoice(outputChoice);
                at_findSdk.execute(m_sub_gray, m_AT_pending, m_sdk);
            } else if (at_findSdk.getStatus() == AsyncTask.Status.FINISHED) {
                m_AT_pending.copyTo(m_AT_finished);
                Mat.zeros(m_AT_pending.rows(), m_AT_pending.cols(), m_AT_pending.type()).copyTo(m_AT_pending);
                at_finished = true;
                sdkFound = at_findSdk.getSdkFound();
                at_findSdk = new AT_findSdk();
            }
        }
    }

    private class AT_findSdk extends AsyncTask<Mat, Void, Void> {
        private boolean at_sdkFound;
        private boolean at_resultChoice;

        protected Void doInBackground(Mat... params){
            at_sdkFound = findSdkCpp(params[0].getNativeObjAddr(), params[1].getNativeObjAddr(), at_resultChoice);
            return null;
        }

        private void setResultChoice(boolean choice){
            at_resultChoice = choice;
        }

        private boolean getSdkFound() {
            return at_sdkFound;
        }
    }

    public boolean onClick(View v){
        Log.d("POLO_D", "onClick - A");

        mOpenCvCameraView.enableView();

        if(sdkFound) {
            sdkFound = false;
        }
        else {
            outputChoice = !outputChoice;
        }

        Log.d("POLO_D", "onClick - B");
        return true;
    }

    void calculateDetectArea(int width, int height){
        Log.d("POLO_D", "calculateDetectArea - A");

        // screen must be landscape oriented !
        final double  inner_square_ratio = 0.1;
        final int     inner_square_or    =(int)Math.floor((double)(height)* inner_square_ratio);

        final int x     = ((width - height) / 2) + inner_square_or;
        final int y     = inner_square_or;
        final int size  = height - (inner_square_or * 2);

        r_detectArea = new Rect(x, y, size, size);

        Log.d("POLO_D", "calculateDetectArea - B");
    }

    void createGuiSdk(){
        // GUI : forme carrée, telle un viseur, servant à définir la zone de detection des sudoku

        // Blank_square : centre vide du rectangle de la GUI
        //                sert à effacer le centre du carré de la GUI
        final double    blank_square_ratio  = 0.05;  // permet de definir la taille du carré qui servira à effacer le centre de la GUI.
                                                    // Plus le nombre est faible, plus les bordures de la GUI sont fines
        final int       blank_square_or     = (int)Math.floor(m_gui_area.size().width * blank_square_ratio);    // Point d'origine du carré
        final int       blank_square_size   = (int)m_gui_area.size().width - (blank_square_or * 2);             // Taille du carré (longueur d'un côté)

        // Blank_rect   : rectangles vide de la GUI
        //                sert à effacer le centre des bords de la GUI
        final double    blank_rect_ratio    = 0.33; // permet de définir la largeur du creux entre les sommets de la GUI.
                                                    // Plus le nombre est faible, plus le creux est étroit.
        final int       blank_rect_or       = (int)Math.floor(m_gui_area.size().width * blank_rect_ratio);      // Point d'origine du rectangle
        final int       blank_rect_size     = (int)m_gui_area.size().width - (blank_rect_or * 2);               // Taille du côté le plus court du rectangle (l'autre étant égal à la taille de GUI

        final Rect blank_square     = new Rect(blank_square_or, blank_square_or, blank_square_size, blank_square_size);
        final Rect blank_rect_ver   = new Rect(blank_rect_or, 0, blank_rect_size, (int)m_gui_area.size().height);
        final Rect blank_rect_hor   = new Rect(0, blank_rect_or, (int)m_gui_area.size().width, blank_rect_size);


        // draw GUI
        // Start with a big square
        final Scalar color = new Scalar(0,255,100);
        m_gui_area.setTo(color);

        // Erase unwanted area to create visor pattern
        Mat dump = new Mat(m_gui_area,blank_square);
        Mat.zeros(dump.size(), dump.type()).copyTo(dump);
        dump = new Mat(m_gui_area,blank_rect_ver);
        Mat.zeros(dump.size(), dump.type()).copyTo(dump);
        dump = new Mat(m_gui_area,blank_rect_hor);
        Mat.zeros(dump.size(), dump.type()).copyTo(dump);
        dump.release();
    }


    public native boolean findSdkCpp(long grayAddr, long outAddr, boolean outParam);

    public native void addWeightedCpp(long m1Addr, long m2Addr, double alpha, double beta);
}
