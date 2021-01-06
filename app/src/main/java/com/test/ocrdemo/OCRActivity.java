package com.test.ocrdemo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.test.ocrdemo.R;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * 1、界面中有检测和识别按钮，点击检测按钮会检测出图片中的文本区域，裁剪为图片并缓存起来<br/>
 * 2、点击识别按钮识别缓存的图片文本<br/>
 * 3、重复点击识别按钮，观察相同输入下，结果却并不是每一次都相同<br/>
 *
 *
 * 未出现异常请多次点击识别按钮<br/>
 *
 * 正常日志如下:
 * <pre>
 * OCRActivity: 0=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 1=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 2=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 3=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 4=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 5=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 6=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 7=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 8=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * OCRActivity: 9=> predicts:[144,08, 144.08, AC01, AFO1, 144.02, 144.05, AB01, AEO1, 144.001, 144.04, AAO1, AD01]
 * </pre>
 */
public class OCRActivity extends AppCompatActivity {

    private final String TAG = "OCRActivity";

    private final String OcrRegModelFileName = "OCR/ocr_reg.mnn";
    private final String OcrDetModelFileName = "OCR/ocr_det.mnn";
    private final String OcrRegWordsFileName = "OCR/alphabet_chinese.txt";
    private final String TargetPic = "OCR/pic.jpg";


    HandlerThread mThread;
    Handler mHandle;

    //检测网络
    private MNNDetNet mDetNet;
    //识别网络
    private MNNRegNet mRegNet;

    private TextView textView;
    private ImageView imageView;
    private Button btnDetect;
    private Button btnReg;
    volatile boolean requestAbort = false;

    static {
        System.loadLibrary("opencv_java3");
    }

    private List<Bitmap> bitmaps;

    private void prepareModels() {
        mDetNet = new MNNDetNet();
        mRegNet = new MNNRegNet();
        //setInputExtend 为true，输入图片等比例缩放后扩展到相同尺寸，避免每一次推理resize inputTensor和resizeSession
        mRegNet.setInputExtend(true);
        mDetNet.setShowLog(false);
        mRegNet.setShowLog(false);
        mDetNet.prepareModels(this, OcrDetModelFileName);
        mRegNet.prepareModels(this, OcrRegModelFileName, OcrRegWordsFileName);
    }

    Bitmap mBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_ocr);
        textView = findViewById(R.id.text);
        imageView = findViewById(R.id.imageView);
        btnDetect = findViewById(R.id.btn_detect);
        btnReg = findViewById(R.id.btn_reg);

        requestAbort = false;
        AssetManager am = getAssets();
        try {
            final InputStream picStream = am.open(TargetPic);
            mBitmap = BitmapFactory.decodeStream(picStream);
            picStream.close();
        } catch (Throwable t) {
            t.printStackTrace();
        }
        // prepare mnn net models
        prepareModels();


        mThread = new HandlerThread("MNNNet-Thread");
        mThread.start();
        mHandle = new Handler(mThread.getLooper());


        btnDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                doDetect();
            }
        });
        btnReg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                doReg();
            }
        });
    }

    private void doDetect() {
        textView.setText("");
        btnDetect.setText("运行中...");
        btnDetect.setEnabled(false);
        btnReg.setEnabled(false);
        btnReg.setText("识别");
        mHandle.post(new Runnable() {
            @Override
            public void run() {
                mDetNet.prepareDetNet();//重新创建推理Session
                doRealDet();
            }
        });
    }


    /**
     * 问题1:
     * <p>
     * 测试每一次识别时都重新创建推理引擎，相同的输入的情况下输出结果会出现不同
     * ,使用64位MNN动态库结果不一致更多
     */
    private void doReg() {
        //还没检测文本，先检测再识别
        if (this.bitmaps == null || this.bitmaps.isEmpty()) {
            doDetect();
            mHandle.post(new Runnable() {
                @Override
                public void run() {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            doReg();
                        }
                    });
                }
            });
            return;
        }
        textView.setText("");
        btnDetect.setText("检测");
        btnReg.setText("运行中...");
        btnDetect.setEnabled(false);
        btnReg.setEnabled(false);
        mHandle.post(new Runnable() {
            @Override
            public void run() {
                mRegNet.prepareNet();//重新创建推理Session
                doRealReg();
            }
        });
    }

    //检测
    private void doRealDet() {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setStrokeWidth(1);
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        List<BoxScore> boxScores = mDetNet.doDet(mBitmap);
        final Bitmap resultBitmap = mBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(resultBitmap);
        for (BoxScore box : boxScores) {
            org.opencv.core.Rect box1 = box.box;
            canvas.drawRect(new Rect(box1.x, box1.y, box1.x + box1.width,
                    box1.y + box1.height), paint);
        }
        bitmaps = mDetNet.getROIs(mBitmap, boxScores);


        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                enableButton();
                imageView.setImageBitmap(resultBitmap);
            }
        });

    }

    private void enableButton() {
        btnDetect.setText("检测");
        btnDetect.setEnabled(true);
        btnReg.setText("识别");
        btnReg.setEnabled(true);
    }

    //识别
    private void doRealReg() {
        final List<String> predicts = new ArrayList<>();
        /*
         * 问题2:
         *  输入图片等比例缩放后扩展到相同尺寸，避免每一次推理resize inputTensor和resizeSession
         * 由com.taobao.android.mnndemo.MNNRegNet#setInputExtend控制是否扩展到相同尺寸,
         * 如果每一次都让resizeSession 多次输入结果也会不一致,甚至出现崩溃
         * 日志如:signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0xbf5dbf70
         */
        int maxWidth = getMaxWidth();
        //测试，重复识别6次
        for (int j = 0; j < 10; j++) {
            if (requestAbort) {
                break;
            }
            predicts.clear();
            for (int i = 0; i < bitmaps.size(); i++) {
                String s = mRegNet.doReg(maxWidth, bitmaps.get(i));
                predicts.add(s);
            }
            Log.d(TAG, j + "=> predicts:" + predicts);
            final List<String> mutableDatas = new ArrayList<>(predicts);
            final int indx = j;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    textView.setText(textView.getText() + "\n" + indx + ":" + mutableDatas);
                }
            });
        }
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                enableButton();
            }
        });
    }

    private int getMaxWidth() {
        int maxWidth = 0;
        for (int i = 0; i < bitmaps.size(); i++) {
            Bitmap bitmap = bitmaps.get(i);
            int width = mRegNet.getInputWidth(bitmap);
            if (maxWidth < width) {
                maxWidth = width;
            }
        }
        return maxWidth;
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        requestAbort = true;
        Log.d(TAG, "--onDestroy--");
        mHandle.post(new Runnable() {
            @Override
            public void run() {
                Log.d(TAG, "---Release MNNNet--");
                mDetNet.release();
                mRegNet.release();
            }
        });
        mThread.quitSafely();

    }
}