package com.test.ocrdemo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 文字检测模型
 */
public class MNNDetNet {
    private static final String TAG = "MNNDetNet";
    //检测网络
    private MNNNetInstance mDetNetInstance;
    private MNNNetInstance.Session mDetSession;
    private MNNNetInstance.Session.Tensor mDetInputTensor;
    private final MNNNetInstance.Config mNetConfig = new MNNNetInstance.Config();// session config

    final MNNImageProcess.Config mImageProcessConfig = new MNNImageProcess.Config();
    /**
     * 输入图片宽度
     */
    private static final int DET_WIDTH = 640;
    /**
     * 输入图片高度
     */
    private static final int DET_HEIGHT = 640;

    private String mDetModelPath;
    private boolean showLog = true;
    private final MNNOCRDetPostProcess postProcess = new MNNOCRDetPostProcess();

    public void setShowLog(boolean showLog) {
        this.showLog = showLog;
    }

    public boolean isShowLog() {
        return showLog;
    }

    public MNNDetNet() {
        // normalization params
        mImageProcessConfig.mean = new float[]{0.485f * 255, 0.456f * 255, 0.406f * 255};
        mImageProcessConfig.normal = new float[]{1f / (255 * 0.229f), 1 / (255 * 0.224f), 1 / (255 * 0.225f)};
        mImageProcessConfig.source = MNNImageProcess.Format.RGBA;// input source format
        mImageProcessConfig.dest = MNNImageProcess.Format.RGB;// input data format
    }

    public void prepareModels(Context context, String OcrDetModelFileName) {
        mDetModelPath = context.getCacheDir() + "ocr_det.mnn";
        try {
            Common.copyAssetResource2File(context, OcrDetModelFileName, mDetModelPath);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public Bitmap getDetectedResult() {
        return postProcess.getDetectedResult();
    }
    /**
     * 创建Session，存在的话先销毁
     * */
    public void prepareDetNet() {
        if (null != mDetSession) {
            mDetSession.release();
            mDetSession = null;
        }
        if (mDetNetInstance != null) {
            mDetNetInstance.release();
            mDetNetInstance = null;
        }

        String modelPath = mDetModelPath;

        // create net instance
        mDetNetInstance = MNNNetInstance.createFromFile(modelPath);

        // mConfig.saveTensors;
        mDetSession = mDetNetInstance.createSession(mNetConfig);

        // get input tensor
        mDetInputTensor = mDetSession.getInput("image");

        int[] dimensions = mDetInputTensor.getDimensions();
        if (showLog) {
            Log.d(TAG, "DetInputTensor dim:" + Arrays.toString(dimensions));
        }
    }

    public List<BoxScore> doDet(Bitmap sourceBitmap) {
        int imageWidth = sourceBitmap.getWidth();
        int imageHeight = sourceBitmap.getHeight();
        copyToInputTensor(sourceBitmap);

        final long startTimestamp = System.nanoTime();
        mDetSession.run();
        MNNNetInstance.Session.Tensor output = mDetSession.getOutput(null);
        int[] dimensions = output.getDimensions();
        if (showLog) {
            Log.d(TAG, "det output tensor dim:" + Arrays.toString(dimensions));
        }
        float[] predictResult = output.getFloatData();// get float results

        float ratio_h = DET_HEIGHT * 1.0f / imageHeight;
        float ratio_w = DET_WIDTH * 1.0f / imageWidth;
        List<BoxScore> boxes = postProcess.process(dimensions, predictResult,
                ratio_h, ratio_w, imageHeight, imageWidth);
        if (showLog) {
            Log.d(TAG, "boxes:" + boxes.size());
        }

        final long endTimestamp = System.nanoTime();
        final float inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f;
        if (showLog) {
            Log.d(TAG, "det cost time：" + inferenceTimeCost + "ms");
        }
        return boxes;
    }

    /**
     * 从原始图片中切割出各个文字区域
     */
    public List<Bitmap> getROIs(Bitmap sourceBitmap, List<BoxScore> boxes) {
        Mat bmp = new Mat();
        Utils.bitmapToMat(sourceBitmap, bmp);
        Imgproc.cvtColor(bmp, bmp, Imgproc.COLOR_RGBA2BGR);
        List<Bitmap> bmps = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            BoxScore box = boxes.get(i);
            Mat mat = MNNOCRDetPostProcess.submat(bmp, box);
            final Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bitmap);
            mat.release();
            bmps.add(bitmap);
        }
        bmp.release();
        return bmps;
    }

    private void copyToInputTensor(Bitmap srcBitmap) {
        Matrix matrix = getTransformMatrix(srcBitmap.getWidth(), srcBitmap.getHeight());
        MNNImageProcess.convertBitmap(srcBitmap, mDetInputTensor, mImageProcessConfig, matrix);
    }

    private Matrix getTransformMatrix(float srcWidth, float srcHeight) {
        // matrix transform: dst to src
        Matrix matrix = new Matrix();
        matrix.postScale(DET_WIDTH / srcWidth,
                DET_HEIGHT / srcHeight);
        matrix.invert(matrix);
        return matrix;
    }

    public void release() {
        if (mDetNetInstance != null) {
            mDetNetInstance.release();
        }
    }
}
