package com.test.ocrdemo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.util.Log;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.TxtFileReader;

import java.util.Arrays;
import java.util.List;

/**
 * 识别模型
 */
public class MNNRegNet {
    private static final String TAG = "MNNRegNet";
    //识别网络
    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    final MNNImageProcess.Config mImageProcessConfig = new MNNImageProcess.Config();

    private final MNNNetInstance.Config mNetConfig = new MNNNetInstance.Config();// session config
    private String mRegModelPath;
    private OcrStrLabelConverter strLabelConverter;
    //输入图片高度
    private static final int REG_INPUT_HEIGHT = 32;
    private boolean showLog = false;

    //输入图片等比例缩放后扩展到相同尺寸
    private boolean inputExtend = true;

    public void setShowLog(boolean showLog) {
        this.showLog = showLog;
    }

    /**
     * @param inputExtend
     * 为true，输入图片等比例缩放后扩展到相同尺寸，避免每一次推理resize inputTensor和resizeSession
     */
    public void setInputExtend(boolean inputExtend) {
        this.inputExtend = inputExtend;
    }

    public boolean isShowLog() {
        return showLog;
    }

    public MNNRegNet() {
        mNetConfig.numThread = 4;
        mNetConfig.forwardType = MNNForwardType.FORWARD_CPU.type;
        // normalization params
        mImageProcessConfig.mean = new float[]{127.5f, 127.5f, 127.5f};
        mImageProcessConfig.normal = new float[]{1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
        mImageProcessConfig.source = MNNImageProcess.Format.RGBA;// input source format
        mImageProcessConfig.dest = MNNImageProcess.Format.GRAY;// input data format
    }

    public void prepareModels(Context context, String OcrRegModelFileName, String OcrRegWordsFileName) {
        mRegModelPath = context.getCacheDir() + "ocr_reg.mnn";
        try {
            Common.copyAssetResource2File(context, OcrRegModelFileName, mRegModelPath);
            List<String> lines = TxtFileReader.getUniqueUrls(context, OcrRegWordsFileName, Integer.MAX_VALUE);
            strLabelConverter = new OcrStrLabelConverter(lines.get(0));
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    /**
     * 创建Session，存在的话先销毁
     * */
    public void prepareNet() {
        if (null != mSession) {
            mSession.release();
            mSession = null;
        }
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }

        String modelPath = mRegModelPath;

        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(modelPath);

        // mConfig.saveTensors;
        mSession = mNetInstance.createSession(mNetConfig);

        // get input tensor
        mInputTensor = mSession.getInput("input");
    }

    private void reshapeInputTensorIfNeed(int inputWidth) {
        int[] dimensions = mInputTensor.getDimensions();
        if (dimensions[0] != 1 || dimensions[3] != inputWidth) {
            dimensions[0] = 1; // force batch = 1  NCHW  [batch, channels, height, width]
            dimensions[3] = inputWidth;
            mInputTensor.reshape(dimensions);
            mSession.reshape();
            if(showLog) {
                Log.d(TAG, "resizeTensor:" + Arrays.toString(mInputTensor.getDimensions()));
            }
        }
    }

    /*
     * 将bitmap等比例缩放后，扩展到相同尺寸
     * */
    private Bitmap extendBitmap(Bitmap bitmap, int targetWidth) {
        Matrix matrix = new Matrix();
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();
        int inputWidth = getInputWidth(height, width);
        matrix.postScale(inputWidth / (float) width,
                REG_INPUT_HEIGHT / (float) height);
        Bitmap bg = Bitmap.createBitmap(targetWidth, REG_INPUT_HEIGHT, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bg);
        canvas.drawColor(Color.BLACK);
        canvas.drawBitmap(bitmap, matrix, null);
        canvas.setBitmap(null);
        return bg;
    }

    public int getInputWidth(Bitmap bitmap) {
        return getInputWidth(bitmap.getHeight(), bitmap.getWidth());
    }

    public int getInputWidth(int height, int width) {
        return Math.round(REG_INPUT_HEIGHT * 1.0f / height * width);
    }

    //识别
    public String doReg(int targetWidth, Bitmap srcBitmap) {
        int inputWidth;
        if (this.inputExtend) {
            srcBitmap = extendBitmap(srcBitmap, targetWidth);
            inputWidth = srcBitmap.getWidth();
        } else {
            inputWidth = getInputWidth(srcBitmap);
        }
        reshapeInputTensorIfNeed(inputWidth);
        copyToInputTensor(srcBitmap, inputWidth);

        final long startTimestamp = System.nanoTime();
        mSession.run();
        MNNNetInstance.Session.Tensor output = mSession.getOutput("out");
        int[] dimensions = output.getDimensions();
//        if (showLog) {
//            Log.d(TAG, "output tensor dim:" + Arrays.toString(dimensions));
//        }
        float[] predictResult = output.getFloatData();// get float results
        final long endTimestamp = System.nanoTime();
        final float inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f;
        if (showLog) {
            Log.d(TAG, "cost time：" + inferenceTimeCost + "ms");
        }
        int[] preds = new int[dimensions[0]];
        for (int i = 0; i < dimensions[0]; i++) {
            int maxValueIdx = 0;
            int flattenBaseOffset = i * dimensions[2];
            for (int k = 0; k < dimensions[2]; k++) {
                int flattenIdx = flattenBaseOffset + k;
                int maxFlattenIdx = flattenBaseOffset + maxValueIdx;
                if (Float.compare(predictResult[flattenIdx], predictResult[maxFlattenIdx]) > 0) {
                    maxValueIdx = k;
                }
            }
            preds[i] = maxValueIdx;
        }
        final String str = strLabelConverter.decode(preds);
        return str;
    }

    private void copyToInputTensor(Bitmap srcBitmap, int regWidth) {
        int imageWidth = srcBitmap.getWidth();
        int imageHeight = srcBitmap.getHeight();
        Matrix matrix = getTransformMatrix(regWidth, imageWidth, imageHeight);
        MNNImageProcess.convertBitmap(srcBitmap, mInputTensor, mImageProcessConfig, matrix);
    }

    private Matrix getTransformMatrix(int regWidth, float srcWidth, float srcHeight) {
        // matrix transform: dst to src
        Matrix matrix = new Matrix();
        matrix.postScale(regWidth / srcWidth,
                REG_INPUT_HEIGHT / srcHeight);
        matrix.invert(matrix);
        return matrix;
    }


    public void release() {
        if (mNetInstance != null) {
            mNetInstance.release();
        }
    }
}
